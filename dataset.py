import os.path

import cv2
import json

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

from settings import IMG_RESIZE
from utils import *


class LSMI(data.Dataset):
    def __init__(self, root, split,
                 input_type='uvl', output_type='uv',
                 illum_augmentation=None, transform=None):
        self.root = root  # dataset root
        self.split = split  # train / val / test
        self.input_type = input_type  # uvl / rgb
        self.output_type = output_type  # None / illumination / uv
        self.random_color = illum_augmentation
        self.transform = transform

        self.image_list = sorted([f for f in os.listdir(os.path.join(root, split))
                                  if f.endswith(".tiff")
                                  and len(os.path.splitext(f)[0].split("_")[-1]) in [1, 2, 3]
                                  and 'gt' not in f])

        meta_file = os.path.join(self.root, 'meta.json')
        with open(meta_file, encoding='utf-8-sig') as meta_json:
            self.meta_data = json.load(meta_json)

        logging.info("[Data]\t" + str(self.__len__()) + " " + split + " images are loaded from " + root)

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_***       : input image (uvl or rgb)
        gt_***          : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse file's name
        filename = os.path.splitext(self.image_list[idx])[0]
        img_file = filename + ".tiff"
        mixture_map = filename + ".npy"
        place, illu_count = filename.split('_')

        # 1. prepare meta information
        ret_dict = {"illu_chroma": np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])}

        for illu_no in illu_count:
            illu_chroma = self.meta_data[place]["Light" + illu_no]
            ret_dict["illu_chroma"][int(illu_no) - 1] = illu_chroma
        ret_dict["img_file"] = img_file
        ret_dict["place"] = place
        ret_dict["illu_count"] = illu_count

        # 2. prepare input & output GT
        # load mixture map & 3 channel RGB tiff image
        input_path = os.path.join(self.root, self.split, img_file)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

        if len(illu_count) != 1:
            mixture_map = np.load(os.path.join(self.root, self.split, mixture_map)).astype('float32')
        else:
            mixture_map = np.ones_like(input_rgb[:, :, 0:1])
        # mixture_map contains -1 for ZERO_MASK, which means uncalculable pixels with LSMI's G channel approximation.
        # So we must replace negative values to 0 if we use pixel level augmentation.
        incalculable_masked_mixture_map = np.where(mixture_map == -1, 0, mixture_map)

        # random data augmentation
        if self.random_color and self.split == 'train':
            augment_chroma = self.random_color(illu_count)
            ret_dict["illu_chroma"] *= augment_chroma
            tint_map = mix_chroma(incalculable_masked_mixture_map, augment_chroma, illu_count)
            input_rgb = input_rgb * tint_map

        ret_dict["input_rgb"] = input_rgb
        ret_dict["input_uvl"] = rgb2uvl(input_rgb)

        # prepare output tensor
        illu_map = mix_chroma(incalculable_masked_mixture_map, ret_dict["illu_chroma"], illu_count)

        ret_dict["gt_illu"] = np.delete(illu_map, 1, axis=2)

        output_bgr = cv2.imread(os.path.join(self.root, self.split, filename + "_gt.tiff"),
                                cv2.IMREAD_UNCHANGED).astype('float32')
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        ret_dict["gt_rgb"] = output_rgb
        output_uvl = rgb2uvl(output_rgb)
        ret_dict["gt_uv"] = np.delete(output_uvl, 2, axis=2)

        # 3. prepare mask
        if self.split == 'train':
            mask = cv2.imread(os.path.join(self.root, self.split, place + '_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = mask[:, :, None].astype('float32')
        else:
            mask = np.ones_like(input_rgb[:, :, 0:1], dtype='float32')

        ret_dict["mask"] = mask
        # 4. apply transform
        if self.transform is not None:
            ret_dict = self.transform(ret_dict)

        return ret_dict

    def __len__(self):
        return len(self.image_list)


class PairedRandomCrop:
    def __init__(self, size=(256, 256), scale=(0.3, 1.0), ratio=(1., 1.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, ret_dict):
        i, j, h, w = RandomResizedCrop.get_params(img=ret_dict['input_rgb'], scale=self.scale, ratio=self.ratio)
        ret_dict['input_rgb'] = TF.resized_crop(ret_dict['input_rgb'], i, j, h, w, self.size)
        ret_dict['input_uvl'] = TF.resized_crop(ret_dict['input_uvl'], i, j, h, w, self.size)
        ret_dict['gt_illu'] = TF.resized_crop(ret_dict['gt_illu'], i, j, h, w, self.size)
        ret_dict['gt_rgb'] = TF.resized_crop(ret_dict['gt_rgb'], i, j, h, w, self.size)
        ret_dict['gt_uv'] = TF.resized_crop(ret_dict['gt_uv'], i, j, h, w, self.size)
        ret_dict['mask'] = TF.resized_crop(ret_dict['mask'], i, j, h, w, self.size)

        return ret_dict


class Resize:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = TF.resize(ret_dict['input_rgb'], self.size)
        ret_dict['input_uvl'] = TF.resize(ret_dict['input_uvl'], self.size)
        ret_dict['gt_illu'] = TF.resize(ret_dict['gt_illu'], self.size)
        ret_dict['gt_rgb'] = TF.resize(ret_dict['gt_rgb'], self.size)
        ret_dict['gt_uv'] = TF.resize(ret_dict['gt_uv'], self.size)
        ret_dict['mask'] = TF.resize(ret_dict['mask'], self.size)

        return ret_dict


class ToTensor:
    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = torch.from_numpy(ret_dict['input_rgb'].transpose((2, 0, 1)))
        ret_dict['input_uvl'] = torch.from_numpy(ret_dict['input_uvl'].transpose((2, 0, 1)))
        ret_dict['gt_illu'] = torch.from_numpy(ret_dict['gt_illu'].transpose((2, 0, 1)))
        ret_dict['gt_rgb'] = torch.from_numpy(ret_dict['gt_rgb'].transpose((2, 0, 1)))
        ret_dict['gt_uv'] = torch.from_numpy(ret_dict['gt_uv'].transpose((2, 0, 1)))
        ret_dict['mask'] = torch.from_numpy(ret_dict['mask'].transpose((2, 0, 1)))

        return ret_dict


class RandomColor:
    def __init__(self, sat_min, sat_max, val_min, val_max, hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self, h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def threshold_test(self, hue_list, hue):
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in illum_count:
            while True:
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(self.sat_min, self.sat_max)
                value = np.random.uniform(self.val_min, self.val_max)
                chroma_rgb = np.array(self.hsv2rgb(hue, saturation, value), dtype='float32')
                chroma_rgb /= chroma_rgb[1]

                if self.threshold_test(hue_list, hue):
                    hue_list.append(hue)
                    ret_chroma[int(i) - 1] = chroma_rgb
                    break

        return np.array(ret_chroma)


def aug_crop():
    tsfm = transforms.Compose([ToTensor(),
                               PairedRandomCrop(size=(IMG_RESIZE, IMG_RESIZE), scale=(0.3, 1.0),
                                                ratio=(1., 1.))])
    return tsfm


def val_resize():

    tsfm = transforms.Compose([ToTensor(), Resize(size=(IMG_RESIZE, IMG_RESIZE))])

    return tsfm