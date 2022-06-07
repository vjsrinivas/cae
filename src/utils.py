import struct
import numpy as np
from torchvision.utils import save_image
import gzip
import numpy as np
import os

def save_imgs(imgs, to_size, name) -> None:
    # x = np.array(x)
    # x = np.transpose(x, (1, 2, 0)) * 255
    # x = x.astype(np.uint8)
    # imsave(name, x)

    # x = 0.5 * (x + 1)

    # to_size = (C, H, W)
    imgs = imgs.clamp(0, 1)
    imgs = imgs.view(imgs.size(0), *to_size)
    print(imgs.shape)
    save_image(imgs, name)

def calculate_size_bottleneck(bottleneck):
    bn_clone = bottleneck.clone()
    bn_clone = bn_clone.int().cpu().numpy()
    np.save('/tmp/tmp.npy', bn_clone)
    with gzip.open('/tmp/tmp.gz', 'wb') as f:
        f.write(bn_clone.tobytes())
    npy_file = os.path.getsize('/tmp/tmp.npy')
    npy_compressed = os.path.getsize('/tmp/tmp.gz')
    return npy_file, npy_compressed

def save_encoded(enc: np.ndarray, fname: str) -> None:
    enc = np.reshape(enc, -1)
    sz = str(len(enc)) + "d"

    with open(fname, "wb") as fp:
        fp.write(struct.pack(sz, *enc))
