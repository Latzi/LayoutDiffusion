"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import json
import os
import time
import imageio
import torch
import torch as th
import numpy as np
from omegaconf import OmegaConf
from torchvision import utils

from layout_diffusion import dist_util, logger
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.respace import build_diffusion
from layout_diffusion.util import fix_seed
from layout_diffusion.dataset.util import image_unnormalize_batch, get_cropped_image
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# ✅ REMOVE Distributed Training (Fix for Colab Single-GPU)
try:
    import torch.distributed as dist
    distributed_available = True
except ImportError:
    distributed_available = False

class FakeDist:
    def get_rank(self): return 0
    def get_world_size(self): return 1
    def init_process_group(self, *args, **kwargs): pass
    def barrier(self): pass

if not distributed_available:
    dist = FakeDist()

def save_image(img_tensor, path):
    """
    Ensures image saving works correctly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if img_tensor is None:
        print(f"❌ ERROR: Image tensor is None. Skipping {path}")
        return

    img_tensor = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img_tensor = (img_tensor * 255).astype(np.uint8)  # Convert to 8-bit image

    imageio.imsave(path, img_tensor)
    print(f"✅ Image saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='./configs/LayoutDiffusion-v1.yaml')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    # ✅ FIX: Skip Distributed Setup in Colab
    logger.log("🚨 Running in SINGLE GPU mode, skipping distributed setup!")

    if cfg.sample.fix_seed:
        fix_seed()

    data_loader = build_loaders(cfg, mode='test')

    total_num_samples = len(data_loader.dataset)
    log_dir = os.path.join(cfg.sample.log_root, 'conditional_{}'.format(cfg.sample.timestep_respacing), 
                           'sample{}x{}'.format(total_num_samples, int(cfg.sample.sample_times)), cfg.sample.sample_suffix)
    logger.configure(dir=log_dir)

    # ✅ FIX: Replace Distributed Logging
    logger.log(f"🚨 Running in SINGLE GPU mode. Total samples: {len(data_loader.dataset)}")
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
            logger.log('✅ Successfully loaded the entire model')
        except:
            logger.log('⚠️ Not successfully loaded the entire model, trying partial load')
            model.load_state_dict(checkpoint, strict=False)

    model.to(dist_util.dev())
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(x, t, obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask, is_valid_obj=is_valid_obj)
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1).to(dist_util.dev())
        obj_class[:, 0] = 0
        obj_bbox = th.zeros_like(obj_bbox).to(dist_util.dev())
        obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1]).to(dist_util.dev())
        is_valid_obj = th.zeros_like(obj_class).to(dist_util.dev())
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask).to(dist_util.dev())
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:]).to(dist_util.dev())

        uncond_image, uncond_extra_outputs = model(x, t, obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask, is_valid_obj=is_valid_obj)
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]  # ✅ Correct return format
        else:
            return mean, {}


    logger.log("creating diffusion...")
    if cfg.sample.sample_method == 'dpm_solver':
        noise_schedule = NoiseScheduleVP(schedule='linear')
    elif cfg.sample.sample_method in ['ddpm', 'ddim']:
        diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)
    else:
        raise NotImplementedError

    logger.log('sample method = {}'.format(cfg.sample.sample_method))
    logger.log("sampling...")

    for batch_idx, batch in enumerate(data_loader):
        print(f'🚨 Running on SINGLE GPU. batch_id={batch_idx}')

        imgs, cond = batch
        imgs = imgs.to(dist_util.dev())

        for sample_idx in range(cfg.sample.sample_times):
            if cfg.sample.sample_method == 'dpm_solver':
                wrappered_model_fn = model_wrapper(model_fn, noise_schedule, is_cond_classifier=False, total_N=1000)
                dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

                x_T = th.randn((imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size), 
                               device=dist_util.dev())
                sample = dpm_solver.sample(x_T, steps=int(cfg.sample.timestep_respacing[0]), eps=float(cfg.sample.eps),
                                           adaptive_step_size=cfg.sample.adaptive_step_size, 
                                           fast_version=cfg.sample.fast_version, clip_denoised=False, rtol=cfg.sample.rtol)
                sample = sample.clamp(-1, 1)
            elif cfg.sample.sample_method in ['ddpm', 'ddim']:
                sample_fn = diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm' else diffusion.ddim_sample_loop
                all_results = sample_fn(
                    model_fn,
                    (imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                    clip_denoised=cfg.sample.clip_denoised,
                    model_kwargs={
                        'obj_class': cond['obj_class'].to(dist_util.dev()) if 'obj_class' in cond else torch.zeros((imgs.shape[0], 1), device=dist_util.dev()),
                        'obj_bbox': cond['obj_bbox'].to(dist_util.dev()) if 'obj_bbox' in cond else torch.zeros((imgs.shape[0], 4), device=dist_util.dev()),
                        'is_valid_obj': cond['is_valid_obj'].to(dist_util.dev()) if 'is_valid_obj' in cond else torch.ones((imgs.shape[0], 1), device=dist_util.dev()),
                        'obj_mask': cond['obj_mask'].to(dist_util.dev()) if 'obj_mask' in cond else torch.zeros_like(imgs, device=dist_util.dev())
                    },
                    cond_fn=None,
                    device=dist_util.dev()
                )
                last_result = all_results[-1]
                sample = last_result['sample'].clamp(-1, 1)
            else:
                raise NotImplementedError

            save_image(sample[0], os.path.join(log_dir, f"generated_image_{sample_idx}.png"))

    logger.log("✅ Sampling complete!")

if __name__ == "__main__":
    main()
