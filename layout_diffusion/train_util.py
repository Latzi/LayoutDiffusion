import copy
import functools
import os

import blobfile as bf
import torch as th
from torch.nn.parallel import DataParallel  # ✅ Use DataParallel instead of DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from layout_diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
import torch
import numpy as np
from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

# For ImageNet experiments, this was a good default value.
INITIAL_LOG_LOSS_SCALE = 20.0
from diffusers.models import AutoencoderKL

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            micro_batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            find_unused_parameters=False,
            only_update_parameters_that_require_grad=False,
            classifier_free=False,
            classifier_free_dropout=0.0,
            pretrained_model_path='',
            log_dir="",
            latent_diffusion=False,
            vae_root_dir="",
            scale_factor=0.18215
    ):
        self.log_dir = log_dir
        logger.configure(dir=log_dir)
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        if pretrained_model_path:
            logger.log(f"loading model from {pretrained_model_path}")
            try:
                model.load_state_dict(
                    th.load(pretrained_model_path, map_location="cpu"), strict=True
                )
            except:
                print('Could not load full model, attempting partial load.')
                model.load_state_dict(
                    th.load(pretrained_model_path, map_location="cpu"), strict=False
                )

        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size if micro_batch_size > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # ✅ Single GPU, no need for world_size

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            only_update_parameters_that_require_grad=only_update_parameters_that_require_grad
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = False  # ✅ Disable DDP
            self.ddp_model = DataParallel(self.model)  # ✅ Use DataParallel
        else:
            logger.warn("Training without GPU may be slow!")
            self.ddp_model = self.model

        self.classifier_free = classifier_free
        self.classifier_free_dropout = classifier_free_dropout
        self.dropout_condition = False

        self.scale_factor = scale_factor
        self.vae_root_dir = vae_root_dir
        self.latent_diffusion = latent_diffusion
        if self.latent_diffusion:
            self.instantiate_first_stage()

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Resume step = {self.resume_step}")
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location="cpu"))

    def run_loop(self):
        def run_loop_generator():
            while (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
            ):
                yield

        for _ in tqdm(run_loop_generator()):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()

            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.micro_batch_size):
            micro = batch[i: i + self.micro_batch_size].to("cuda")  # ✅ Force GPU usage
            if self.latent_diffusion:
                micro = self.get_first_stage_encoding(micro).detach()
            micro_cond = {
                k: v[i: i + self.micro_batch_size].to("cuda")
                for k, v in cond.items() if k in self.model.layout_encoder.used_condition_types
            }
            last_batch = (i + self.micro_batch_size) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], "cuda")

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def save(self):
        save_path = os.path.join(self.log_dir, f"model{self.step:07d}.pt")
        th.save(self.model.state_dict(), save_path)
        logger.log(f"✅ Model saved at {save_path}")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
