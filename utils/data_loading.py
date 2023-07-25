import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
import cv2


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(
            f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):

    def __init__(self,
                 images_dir: str,
                 mask_dir: str,
                 scale: float = 1.0,
                 mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0] for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there'
            )

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(
                tqdm(p.imap(
                    partial(unique_mask_values,
                            mask_dir=self.mask_dir,
                            mask_suffix=self.mask_suffix), self.ids),
                     total=len(self.ids)))

        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        self.transform = A.Compose([
            A.Flip(p=0.2),
            A.ToGray(p=0.2),
            A.ColorJitter(p=0.2,
                          brightness=0.2,
                          contrast=0.2,
                          saturation=0.2,
                          hue=0.1),
            A.OneOf([
                A.Blur(p=0.2, blur_limit=(3, 5)),
                A.MedianBlur(p=0.2, blur_limit=(3, 5))
            ]),
            A.OneOf([A.Perspective(p=0.2),
                     A.Rotate(p=0.2, limit=10)]),
            A.CoarseDropout(p=0.2)
        ])

        self.target_shape = (512, 512)

    def __len__(self):
        return len(self.ids)

    def padding_to_square(self,
                          img: np.ndarray,
                          target_shape: tuple,
                          padding_val: int = 127) -> np.ndarray:

        # 1. 計算填充邊界所需要的空間
        h_target, w_target = self.target_shape
        new_h, new_w = img.shape[:2]
        top = int((h_target - new_h) / 2)
        bottom = h_target - new_h - top
        left = int((w_target - new_w) / 2)
        right = w_target - new_w - left

        #  2. 填充邊界
        v = (padding_val, padding_val, padding_val)
        img_filled = cv2.copyMakeBorder(img,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv2.BORDER_CONSTANT,
                                        value=v)

        return img_filled

    def preprocess(self, mask_values, pil_img, pil_mask):

        # 1. PIL 轉 OpenCV ndarray
        np_img = np.asarray(pil_img)
        np_mask = np.asarray(pil_mask)

        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        np_mask = cv2.cvtColor(np_mask, cv2.COLOR_RGB2BGR)

        # 2. 資料增強
        transformed = self.transform(image=np_img, mask=np_mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        # BGR to RGB
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        transformed_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB)
        
        # 3. 填充邊界
        img_filled = self.padding_to_square(transformed_image,
                                            self.target_shape)
        mask_filled = self.padding_to_square(transformed_mask,
                                             self.target_shape, 0)

        # 4. 整理資料
        # 取得資料
        h_target, w_target = self.target_shape

        # 遮罩
        mask = np.zeros((h_target, w_target), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if mask_filled.ndim == 2:
                mask[mask_filled == v] = i
            else:
                mask[(mask_filled == v).all(-1)] = i

        # 影像
        if img_filled.ndim == 2:
            img_filled = img_filled[np.newaxis, ...]
        else:
            img_filled = img_filled.transpose((2, 0, 1))

        if (img_filled > 1).any():
            img_filled = img_filled / 255.0

        return img_filled, mask

    def __getitem__(self, idx):

        # 1. 計算檔名
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        # 2. 檢查數量
        assert len(
            img_file
        ) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file
        ) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        # 3. 讀取照片
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # 檢查照片跟標記尺寸是否一樣
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(self.mask_values, img, mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):

    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
