import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import transforms
import cv2
import os
from itertools import islice

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
warnings.filterwarnings("ignore")
from torchvision.utils import make_grid
from PIL import Image
from einops import rearrange
import pickle
import pandas as pd
from utils.template import FORMAT, Templates, CELLTYPE_FORMAT, celltype_Templates
from multiprocessing.dummy import Pool as ThreadPool

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model(config_path, device, checkpoint):
    config = OmegaConf.load(config_path)
    # del config['model']['params']['first_stage_config']['params']['ckpt_path']

    model = load_model_from_config(config, checkpoint, device)
    return model

def get_unconditional_token(batch_size):
    return [""]*batch_size

def get_conditional_token(batch_size, summary):
    # append tumor and TIL probability to the summary
    # tumor = "high grade; "
    return [summary for _ in range(batch_size)]

def gen_baseline_st_sample(sampler, batch, img_shape, st_shape, scale):
    batch_size = batch['conds']['image'].shape[0]
    with torch.no_grad():
        # w celltype inference
        batch['conds']['image'] = batch['unconds']['image']
        batch['conds']['celltype'] = batch['unconds']['celltype']
        ct = batch['conds']
        cc = model.get_learned_conditioning(ct)
        ut = batch['unconds']
        uc = model.get_learned_conditioning(ut)
        samples_ddim_sts, _ = sampler.sample(50, batch_size, shape=st_shape, conditioning=cc, verbose=False, \
                                         unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=0)

        samples_ddim_sts = samples_ddim_sts.cpu().numpy()

    return samples_ddim_sts


if __name__ == "__main__":
    from infer.args_parser import parser
    args = parser.parse_args()
    sample_by_cls = args.sample_by_cls
    tar_organ = args.tar_organ
    tar_cancer_type = args.tar_cancer_type
    img_shape = args.img_shape
    st_shape = args.st_shape
    exp_name = args.exp_name
    scale = args.scale
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    split = args.split
    device = torch.device(args.device)
    results_save_dir = os.path.join(args.results_save_dir, exp_name)

    os.makedirs(results_save_dir, exist_ok=True)
    from ldm.data.joint_diff.st_emb_label_cellanno_aug_dataset import STEmbDataset
    from tqdm import tqdm
    config = {
        "root": "/disk2/tma/VISTA/hest_benchmark_visium_v5",
        "split": args.split,
        'prompts_file': None,
        'emb_path': args.emb_path,
        'crop_size': 256,
        'p_uncond': 0.1,
        'sample': True,
        'use_celltype': True,
        'use_max_cell': False,
    }

    dataset = STEmbDataset(config=config)
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = get_model(config_path, device=args.device, checkpoint=ckpt_path)

    sampler = DDIMSampler(model)

    gen_genes = []
    for b_idx, batch in tqdm(enumerate(test_dataloader)):
        samples_ddim_sts = gen_baseline_st_sample(sampler, batch, img_shape, st_shape, scale,)
        gen_genes.append(samples_ddim_sts)

    gen_genes = np.concatenate(gen_genes, axis=0)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    np.save(os.path.join(results_save_dir, f'mixcontrol-st-embs-wo-img-{split}-wo-celltype-{scale}.npy'), gen_genes)















