from share import *
from torch.utils.data.dataset import ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model_loader import create_model, load_state_dict
from train_util.multi_task_scheduler import BatchSchedulerSampler
from train_util.dataset import AIRDataset, BasicImageDataset

import json
import argparse
import torch

torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help='path to SD1.5 checkpoint to initialise from')
parser.add_argument("--config", type=str, default='./models/config.yaml')
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--img_logger_freq", type=int, default=607)
parser.add_argument("--val_img_logger_freq", type=int, default=15013)
parser.add_argument("--ckpt_logger_freq", type=int, default=15000)
parser.add_argument("--out_path", type=str, default='./output', help='logging / dataset root path')
parser.add_argument("--ckpt_out_path", type=str, default='./checkpoints')
parser.add_argument("--task_prompts", type=str, default='', help='optional JSON task prompt bank')
args = parser.parse_args()

model = create_model(args.config).cpu()
model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=True)
torch.cuda.empty_cache()

vae_ckpt_path = "/work/users/d/e/debman/air-diffusion-ckpts/ckpts/vae-ft-mse-840000-ema-pruned.ckpt"
vae_ckpt = torch.load(vae_ckpt_path, map_location="cuda", weights_only=False)
vae_state_dict = vae_ckpt.get("state_dict", vae_ckpt.get("first_stage_model", vae_ckpt))
vae_state_dict = {k.replace("first_stage_model.", ""): v for k, v in vae_state_dict.items()}
model.first_stage_model.load_state_dict(vae_state_dict, strict=False)

model.learning_rate = args.lr
model.sd_locked = True
model.only_mid_control = False

# ── Dataset paths ──────────────────────────────────────────────────────────────
defocus_train = '/work/users/d/e/debman/degradations_refined_final/defocus_blur/Defocus/defocus_formal/train/blurry'
defocus_test  = '/work/users/d/e/debman/degradations_refined_final/defocus_blur/Defocus/defocus_formal/test/blurry'
SOTS_indoor   = '/work/users/d/e/debman/degradations_refined_final/haze/reside/SOTS/indoor/hazy'
SOTS_outdoor  = '/work/users/d/e/debman/degradations_refined_final/haze/reside/SOTS/outdoor/hazy'
ots_hazy      = '/work/users/d/e/debman/degradations_refined_final/haze/reside/ots/hazy'
cdd_haze      = '/work/users/d/e/debman/degradations_refined_final1/cdd/train/CDD-11_train/haze'
lol_train     = '/work/users/d/e/debman/degradations_refined_final/lwlight/lol/LOLdataset/our485/low'
lol_eval      = '/work/users/d/e/debman/degradations_refined_final/lwlight/lol/LOLdataset/eval15/low'
lol_blur      = '/work/users/d/e/debman/degradations_refined_final/lwlight/lol-blur1/train/low_sharp'
div2k_noise   = '/work/users/d/e/debman/degradations_refined_final/noise/div2k/DIV2K_train_HR'
noise_extra   = '/work/users/d/e/debman/degradations_refined_final1/noise'

# task_id: 4-bit binary vector selecting which of the 4 task heads to activate
# control: unused here (kept for reference); head routing is via task_id alone
task_id_map = {
    'defocus deblur':        [1, 0, 0, 0],
    'haze removal':          [0, 1, 0, 0],
    'low light enhancement': [0, 0, 1, 0],
    'denoise':               [0, 0, 0, 1],
}

train_groups = [
    ([defocus_train],                    'defocus deblur'),
    ([SOTS_indoor, ots_hazy, cdd_haze],  'haze removal'),
    ([lol_train, lol_blur],              'low light enhancement'),
    ([div2k_noise, noise_extra],         'denoise'),
]
val_groups = [
    ([defocus_test],  'defocus deblur'),
    ([SOTS_outdoor],  'haze removal'),
    ([lol_eval],      'low light enhancement'),
]


def make_dataset(paths, task_name, is_train):
    return AIRDataset(
        BasicImageDataset(paths, is_train=is_train),
        task_id=task_id_map[task_name],
        task_name=task_name,
        train_unconditional_guidance=is_train,
    )


train_datasets = [make_dataset(paths, name, True)  for paths, name in train_groups]
val_datasets   = [make_dataset(paths, name, False) for paths, name in val_groups]

for ds in train_datasets + val_datasets:
    print(len(ds))

accum_rate = 16
multi_dataset_tr  = ConcatDataset(train_datasets)
multi_dataset_val = ConcatDataset(val_datasets)

dataloader_tr = DataLoader(
    multi_dataset_tr,
    num_workers=8,
    sampler=BatchSchedulerSampler(dataset=multi_dataset_tr, batch_size=args.bs, permute=False, accum=accum_rate),
    batch_size=args.bs,
    persistent_workers=True,
    shuffle=False,
)
dataloader_val = DataLoader(multi_dataset_val, num_workers=2, batch_size=args.bs, persistent_workers=True, shuffle=True)

logger_img        = ImageLogger(batch_frequency=args.img_logger_freq)
logger_metrics    = CSVLogger(args.out_path, name="metrics.csv")
logger_checkpoint = CheckpointEveryNSteps(save_step_frequency=args.ckpt_logger_freq, dirpath=f"{args.ckpt_out_path}_step")
logger_checkpoint_auto = ModelCheckpoint(
    dirpath=f"{args.ckpt_out_path}_best",
    every_n_train_steps=1024,
    monitor='train/loss_latent',
    save_top_k=6,
)

trainer = pl.Trainer(
    default_root_dir=args.out_path,
    devices=args.gpus,
    precision='bf16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[logger_checkpoint, logger_checkpoint_auto, logger_img],
    logger=logger_metrics,
    accelerator="gpu",
    use_distributed_sampler=False,
    accumulate_grad_batches=accum_rate,
    val_check_interval=args.val_img_logger_freq,
    limit_val_batches=4,
)

if __name__ == '__main__':
    trainer.fit(model, dataloader_tr, dataloader_val)
