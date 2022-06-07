import os
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import SingleImage, preprocess_single
from utils import save_imgs

from namespace import Namespace
from logger import Logger

from models.cae_32x32x32_zero_pad_bin import CAE
import cv2

ROOT_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"

logger = Logger(__name__, colorize=True)


def detect(cfg: Namespace, image, show_stats=False) -> None:
    assert cfg.checkpoint not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    exp_dir = ROOT_EXP_DIR / cfg.exp_name
    os.makedirs(exp_dir / "out", exist_ok=True)
    cfg.to_file(exp_dir / "test_config.json")
    logger.info(f"[exp dir={exp_dir}]")

    model = CAE(cfg, check_size=show_stats)
    model.load_state_dict(torch.load(cfg.checkpoint))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()
    logger.info(f"[model={cfg.checkpoint}] on {cfg.device}")

    img, patches, pad_img, pad = preprocess_single(image)
    if cfg.device == "cuda":
        patches = patches.cuda()

    #out = T.zeros(6, 10, 3, 128, 128)
    ps = patches.shape
    out = torch.zeros(ps[1], ps[2], ps[0], ps[3], ps[4])

    for i in range(pad[1]):
        for j in range(pad[0]):
            x = patches[:, i, j, :, :]
            x = torch.unsqueeze(x, axis=0)
            if cfg.device == "cuda":
                x.cuda()
            y = model(x)[0]
            out[i, j] = y.data
    
    # save output
    out = np.transpose(out, (0, 3, 1, 4, 2))
    out = np.reshape(out, (pad_img[0], pad_img[1], 3))
    out = np.transpose(out, (2, 0, 1))

    y = torch.cat((img, out), dim=2)

    if show_stats:
        print("Original size (bytes): %i"%(model.original_file_size))
        print("Compressed size (bytes): %i"%(model.compressed_file_size))

    save_imgs(
        imgs=y.unsqueeze(0),
        to_size=(3, pad_img[0], 2 * pad_img[1]),
        name=exp_dir / f"out/custom.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--size", action='store_true')
    args = parser.parse_args()

    with open(args.config, "rt") as fp:
        cfg = Namespace(**yaml.safe_load(fp))

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detect(cfg, image, show_stats=args.size)
