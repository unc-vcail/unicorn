import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from train_util.hasher import hash_state_dict


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if k == "hints_input":
                images[k] = torch.cat([_k.expand(_k.shape[0], 3, _k.shape[2], _k.shape[3]) for _k in images[k]], dim=0)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
            grid = (grid * 255).astype(np.uint8)[..., :3]
            filename = "{}_gs-{:08}_e-{:08}_b-{:08}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) or split == "val") and \
                hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            _images = []
            for k in images:
                if k == "hints_input":
                    for _k in images[k]:
                        if _k is not None:
                            _images.append(_k)
                    images[k] = _images
                    break

            for k in images:
                N = min(images[k][0].shape[0], self.max_images)
                images[k] = [_k[:N] for _k in images[k]]
                if isinstance(images[k][0], torch.Tensor):
                    images[k] = [_k.detach().cpu() for _k in images[k]]
                    if self.clamp:
                        images[k] = [torch.clamp(_k, -1., 1.) for _k in images[k]]

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            with torch.no_grad():
                self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled:
            with torch.no_grad():
                self.log_img(pl_module, batch, batch_idx, split="val")


class CheckpointEveryNSteps(Callback):
    """Saves a checkpoint every N training steps."""

    def __init__(self, save_step_frequency, prefix="checkpoint", use_modelcheckpoint_filename=False, dirpath=""):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath
        self.last_saved_step = -1
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step > 0 and global_step % self.save_step_frequency == 0 and global_step != self.last_saved_step:
            model_hash = hash_state_dict(pl_module.state_dict())
            print(f"Saving checkpoint at step {global_step}, hash {model_hash}")
            epoch = trainer.current_epoch
            filename = (trainer.checkpoint_callback.filename if self.use_modelcheckpoint_filename
                        else f"{self.prefix}_epoch={epoch}_step={global_step}.ckpt")
            trainer.save_checkpoint(os.path.join(self.dirpath, filename))
            self.last_saved_step = global_step
