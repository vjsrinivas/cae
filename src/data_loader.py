from pathlib import Path
from typing import Tuple

import numpy as np
import torch as T
from PIL import Image
from torch.utils.data import Dataset

def preprocess_single(img):
    pad_w, pad_h = int(np.ceil(img.shape[1]/128)), int(np.ceil(img.shape[0]/128))
    leftover_w, leftover_h = ((pad_w*128)-img.shape[1]), ((pad_h*128)-img.shape[0])
    if leftover_w%2 != 0:
        w_1 = leftover_w//2
        w_2 = leftover_w-w_1
    else:
        w_1 = leftover_w//2
        w_2 = w_1
    if leftover_h%2 != 0:
        h_1 = leftover_h//2
        h_2 = leftover_w-h_1
    else:
        h_1 = leftover_h//2
        h_2 = h_1

    pad = ((h_1,h_2), (w_1,w_2), (0,0))
    img = np.pad(img, pad, mode="edge") / 255.0
    pad_img_new = img.shape

    img = np.transpose(img, (2, 0, 1))
    img = T.from_numpy(img).float()

    patches = np.reshape(img, (3, pad_h, 128, pad_w, 128))
    patches = np.transpose(patches, (0, 1, 3, 2, 4))
    return img, patches, pad_img_new, (pad_w, pad_h)


class SingleImage(Dataset):
    def __init__(self, path):
        self.path = path

    def __getitem__(self, index):
        img = np.array(Image.open(self.path))
        
        pad_w, pad_h = int(np.ceil(img.shape[1]/128)), int(np.ceil(img.shape[0]/128))
        leftover_w, leftover_h = ((pad_w*128)-img.shape[1]), ((pad_h*128)-img.shape[0])
        if leftover_w%2 != 0:
            w_1 = leftover_w//2
            w_2 = leftover_w-w_1
        else:
            w_1 = leftover_w//2
            w_2 = w_1
        if leftover_h%2 != 0:
            h_1 = leftover_h//2
            h_2 = leftover_w-h_1
        else:
            h_1 = leftover_h//2
            h_2 = h_1

        pad = ((h_1,h_2), (w_1,w_2), (0,0))
        img = np.pad(img, pad, mode="edge") / 255.0
        pad_img_new = img.shape

        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, pad_h, 128, pad_w, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        return img, patches, self.path, pad_img_new, (pad_w, pad_h)

    def __len__(self):
        return 1

class ImageFolder720p(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """

    def __init__(self, root: str):
        self.files = sorted(Path(root).iterdir())

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, str]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path))
        pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        img = np.pad(img, pad, mode="edge") / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, 6, 128, 10, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        return img, patches, path

    def __len__(self):
        return len(self.files)
