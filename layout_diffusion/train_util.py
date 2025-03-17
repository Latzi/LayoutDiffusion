import copy
import os

import blobfile as bf
import torch as th
import torch
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from layout_diffusion.resample import UniformSampler
from tqdm import tqdm
import numpy as np
from diffusers.models import AutoencoderKL

INITIAL_LOG_LOSS_SCALE = 20.0

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
                    dist_util.load_state_dict(pretrained_model_path, map_location="cpu"), strict=True
                )
            except:
                print('Could not load full model, attempting partial load.')
                model.load_state_dict(
                    dist_util.load_state_dict(pretrained_model_path, map_location="cpu"), strict=False
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
        self.global_batch = self.batch_size

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

        self.use_ddp = False
        self.ddp_model = self.model

        self.classifier_free = classifier_free
        self.classifier_free_dropout = classifier_free_dropout

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
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def run_loop(self):
        print(f"🚀 Training started with batch size {self.batch_size}, learning rate {self.lr}")

        for _ in tqdm(range(self.lr_anneal_steps or 1_000_000)):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

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
        micro = batch.to(dist_util.dev())

        micro_cond = {k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v for k, v in cond.items() if k != "obj_class_name"}

        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
        losses = self.diffusion.training_losses(self.model, micro, t, model_kwargs=micro_cond)
        loss = (losses["loss"] * weights).mean()

        # ✅ Compute gradients
        grad_norm = th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # ✅ Save grad_norm and other parameters only on save steps
        if self.step % self.save_interval == 0 and self.step > 0:
            print("\n----------------------------")
            print(f"| step          | {self.step:<8} |")
            print(f"| grad_norm     | {grad_norm:<8.2f} |")
            print(f"| lg_loss_scale | {INITIAL_LOG_LOSS_SCALE:<8} |")
            print(f"| param_norm    | {sum(p.norm().item() for p in self.model.parameters() if p is not None):<8.2f} |")
            print(f"| samples       | {self.step * self.batch_size:<8} |")
            print("----------------------------\n")

        self.mp_trainer.backward(loss)

    def save(self):
        try:
            save_path = os.path.join(self.log_dir, f"model{self.step:07d}.pt")
            print(f"💾 DEBUG: Preparing to save model at step {self.step}...")

            print("🔍 DEBUG: Checking model parameters before saving...")
            
            # ✅ Extract state_dict from model
            raw_state_dict = self.model.state_dict()
            print(f"✅ DEBUG: Extracted raw state_dict.")

            # 🚨 Check raw_state_dict before conversion
            found_issue = False
            for param_name, param_value in raw_state_dict.items():
                if not isinstance(param_value, th.Tensor):
                    print(f"🚨 ERROR: {param_name} is {type(param_value)} instead of Tensor!")
                    found_issue = True

            if found_issue:
                raise ValueError("❌ ERROR: At least one parameter in raw_state_dict is invalid!")

            print("🔍 DEBUG: Converting state_dict using master_params_to_state_dict...")

            # ✅ Wrap the conversion in a try-except block
            try:
                state_dict = self.mp_trainer.master_params_to_state_dict(raw_state_dict)
                print("✅ DEBUG: Successfully converted state_dict!")
            except Exception as e:
                print(f"🔥 ERROR INSIDE master_params_to_state_dict(): {e}")
                print(f"🔍 DEBUG: Checking which parameter is causing the issue...")

                for param_name, param_value in raw_state_dict.items():
                    try:
                        _ = param_value.view(-1)  # Try using `.view()` on the parameter
                    except AttributeError:
                        print(f"🚨 ERROR: {param_name} is not a tensor, but a {type(param_value)}")

                raise ValueError("🔥 master_params_to_state_dict() failed due to invalid parameter.")

            print(f"💾 DEBUG: Saving model to {save_path}...")
            with open(save_path, "wb") as f:
                th.save(state_dict, f)

            print(f"✅ Model successfully saved at: {save_path}")

        except Exception as e:
            print(f"🚨 ERROR DURING SAVE: {e} | Skipping save but continuing training.")

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

