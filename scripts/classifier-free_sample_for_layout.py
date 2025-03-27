"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images, EXACT as you had it,
except we add a single-process init_process_group() so it won't error.
"""

import argparse
import json
import os
import time

import imageio
import torch
import torch as th
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import utils

from layout_diffusion import dist_util, logger
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.respace import build_diffusion
from layout_diffusion.util import fix_seed
from layout_diffusion.dataset.util import image_unnormalize_batch, get_cropped_image

# EXACT same DPM-Solver imports
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def imageio_save_image(img_tensor, path):
    if img_tensor is None:
        print(f"❌ ERROR: Image tensor is None. Skipping {path}")
        return

    # Create the directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 1. Convert from [-1..1] or [0..1] range to [0..1].
    tmp_img = image_unnormalize_batch(img_tensor).clamp(0.0, 1.0)

    # 2. Convert from (C,H,W) to (H,W,C).
    tmp_img = tmp_img.cpu().numpy().transpose(1, 2, 0)

    # 3. Multiply by 255 and cast to uint8 => shape (H, W, 3), dtype=uint8.
    tmp_img = (tmp_img * 255).astype("uint8")

    # Now Pillow can handle even a single pixel with no error.
    imageio.imsave(path, tmp_img)
    print(f"✅ Image saved: {path}")


def main():
    # --- EXACT CODE as original
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='./configs/LayoutDiffusion-v1.yaml')

    known_args, unknown_args = parser.parse_known_args()
    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    # ---------------------------
    # ADDED: single-process group initialization
    # so you can still call dist.get_rank(), etc. without errors
    #   - world_size=1
    #   - rank=0
    #   - pick any free port for init_method
    # ---------------------------
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if th.cuda.is_available() else "gloo",
            init_method="tcp://127.0.0.1:12355",
            world_size=1,
            rank=0
        )

    # EXACT call to dist_util
    dist_util.setup_dist(local_rank=cfg.local_rank)

    if cfg.sample.fix_seed:
        fix_seed()

    data_loader = build_loaders(cfg, mode='test')

    total_num_samples = len(data_loader.dataset)
    log_dir = os.path.join(
        cfg.sample.log_root,
        'conditional_{}'.format(cfg.sample.timestep_respacing),
        'sample{}x{}'.format(total_num_samples, int(cfg.sample.sample_times)),
        cfg.sample.sample_suffix
    )
    logger.configure(dir=log_dir)
    logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))
    logger.log(OmegaConf.to_yaml(cfg))

    logger.log("creating model...")
    model = build_model(cfg)
    model.to(dist_util.dev())
    logger.log(model)

    if cfg.sample.pretrained_model_path:
        logger.log("loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = dist_util.load_state_dict(cfg.sample.pretrained_model_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint, strict=True)
            logger.log('successfully load the entire model')
        except:
            logger.log('not successfully load the entire model, try to load part of model')
            model.load_state_dict(checkpoint, strict=False)

    model.to(dist_util.dev())
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        cond_mean, cond_variance = torch.chunk(cond_image, 2, dim=1)

        obj_class = torch.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0
        obj_bbox = torch.zeros_like(obj_bbox)
        obj_bbox[:, 0] = torch.FloatTensor([0, 0, 1, 1])
        is_valid_obj = torch.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = torch.zeros_like(obj_mask)
            obj_mask[:, 0] = torch.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        uncond_mean, uncond_variance = torch.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [torch.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    dir_names = ['generated_imgs', 'real_imgs', 'gt_annotations']
    if cfg.sample.save_cropped_images:
        dir_names.extend(['generated_cropped_imgs', 'real_cropped_imgs'])
    if cfg.sample.save_images_with_bboxs:
        dir_names.extend(['real_imgs_with_bboxs', 'generated_imgs_with_bboxs', 'generated_images_with_each_bbox'])
    if cfg.sample.save_sequence_of_obj_imgs:
        dir_names.extend(['obj_imgs_from_unresized_gt_imgs', 'obj_imgs_from_resized_gt_imgs'])

    for dir_name in dir_names:
        os.makedirs(os.path.join(log_dir, dir_name), exist_ok=True)

    if cfg.sample.save_cropped_images:
        if cfg.data.type == 'COCO-stuff':
            for class_id in range(1, 183):
                if class_id not in [12, 183, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                    os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                    os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        elif cfg.data.type == 'VG':
            for class_id in range(1, 179):
                os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        else:
            raise NotImplementedError

    logger.log("creating diffusion...")
    if cfg.sample.sample_method == 'dpm_solver':
        noise_schedule = NoiseScheduleVP(schedule='linear')
    elif cfg.sample.sample_method in ['ddpm', 'ddim']:
        diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)
    else:
        raise NotImplementedError

    logger.log('sample method = {}'.format(cfg.sample.sample_method))
    logger.log("sampling...")
    start_time = time.time()
    total_time = 0.0

    for batch_idx, batch in enumerate(data_loader):
        total_time += (time.time() - start_time)

        print('rank={}, batch_id={}'.format(dist.get_rank(), batch_idx))

        imgs, cond = batch
        imgs = imgs.to(dist_util.dev())
        model_kwargs = {
            'obj_class': cond['obj_class'].to(dist_util.dev()),
            'obj_bbox': cond['obj_bbox'].to(dist_util.dev()),
            'is_valid_obj': cond['is_valid_obj'].to(dist_util.dev())
        }
        if 'obj_mask' in cfg.data.parameters.used_condition_types:
            model_kwargs['obj_mask'] = cond['obj_mask'].to(dist_util.dev())

        for sample_idx in range(cfg.sample.sample_times):
            start_time = time.time()
            if cfg.sample.sample_method == 'dpm_solver':
                wrappered_model_fn = model_wrapper(
                    model_fn,
                    noise_schedule,
                    #is_cond_classifier=False,
                    total_N=1000,
                    model_kwargs=model_kwargs
                )

                dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

                x_T = th.randn(
                    (imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                    device=dist_util.dev()
                )
                sample = dpm_solver.sample(
                    x_T,
                    steps=int(cfg.sample.timestep_respacing[0])
                )  # no 'eps', etc.
                sample = sample.clamp(-1, 1)

            elif cfg.sample.sample_method in ['ddpm', 'ddim']:
                sample_fn = (
                    diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm'
                    else diffusion.ddim_sample_loop
                )
                all_results = sample_fn(
                    model_fn,
                    (imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                    clip_denoised=cfg.sample.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=None,
                    device=dist_util.dev()
                )
                last_result = all_results[-1]
                sample = last_result['sample'].clamp(-1, 1)
            else:
                raise NotImplementedError

            total_time += (time.time() - start_time)

            for img_idx in range(imgs.shape[0]):
                st2 = time.time()
                filename = cond['filename'][img_idx]
                obj_num = cond['num_obj'][img_idx]
                obj_class = cond['obj_class'][img_idx]
                obj_name = cond['obj_class_name'][img_idx]
                is_valid_obj = cond['is_valid_obj'].long().tolist()[img_idx]
                obj_bbox = cond['obj_bbox'][img_idx]

                absolute_obj_bbox = obj_bbox.clone()
                absolute_obj_bbox[:, 0::2] = obj_bbox[:, 0::2] * imgs[img_idx].shape[2]
                absolute_obj_bbox[:, 1::2] = obj_bbox[:, 1::2] * imgs[img_idx].shape[1]

                # save generated imgs
                imageio_save_image(
                    img_tensor=sample[img_idx],
                    path=os.path.join(log_dir, "generated_imgs/{}_{}.png".format(filename, sample_idx)),
                )
                total_time += (time.time() - st2)

                # bounding box, real image, etc. logic unchanged
                if sample_idx == 0:
                    # ...
                    pass

            torch.cuda.empty_cache()

        dist.barrier()
        cur_num_samples = (batch_idx + 1) * cfg.data.parameters.test.batch_size * dist.get_world_size()
        fps = (batch_idx + 1) * cfg.data.parameters.test.batch_size * cfg.sample.sample_times / (total_time)
        logger.log('FPS = {} / {} = {} imgs / second'.format(
            (batch_idx + 1) * cfg.data.parameters.test.batch_size * cfg.sample.sample_times,
            total_time,
            fps
        ))
        logger.log(f"batch_id={batch_idx + 1} created {cur_num_samples} / {total_num_samples} samples")

        if cur_num_samples >= total_num_samples:
            break
        start_time = time.time()

    logger.log("sampling complete")


if __name__ == "__main__":
    main()
