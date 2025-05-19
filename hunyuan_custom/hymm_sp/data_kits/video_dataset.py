import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from hymm_sp.data_kits.data_tools import *


class DataPreprocess(object):
    def __init__(self):
        self.llava_size = (336, 336)
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def get_batch(self, image_path, size):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(image_path).convert('RGB')
        llava_item_image = pad_image(image.copy(), self.llava_size)
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        cat_item_image = pad_image(image.copy(), size)

        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0
        batch = {
            "pixel_value_llava": llava_item_tensor.unsqueeze(0),
            "uncond_pixel_value_llava": uncond_llava_item_tensor.unsqueeze(0),
            'pixel_value_ref': cat_item_tensor.unsqueeze(0), 
        }
        return batch


class JsonDataset(object):
    def __init__(self, args):
        self.args = args
        self.data_list = args.input
        self.pad_color = (255, 255, 255)
        self.llava_size = (336, 336)
        self.ref_size = (args.video_size[1], args.video_size[0])
        if self.data_list.endswith('.list'):
            self.data_paths = [line.strip() for line in open(self.data_list, 'r')] if self.data_list is not None else []
        else:
            self.data_paths = [self.data_list]
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        
    def __len__(self):
        return len(self.data_paths)
    
    def read_image(self, image_path):
        if isinstance(image_path, dict):
            image_path = image_path['seg_item_image_path']

        try:
            face_image_masked = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        except:
            face_image_masked = Image.open(image_path).convert('RGB')

        cat_face_image = pad_image(face_image_masked.copy(), self.ref_size)
        llava_face_image = pad_image(face_image_masked.copy(), self.llava_size)
        return llava_face_image, cat_face_image

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data_name = os.path.basename(os.path.splitext(data_path)[0])
        if data_path.endswith('.json'):
            data = json.load(open(data_path, 'r'))
            llava_item_image, cat_item_image = self.read_image(data)
            item_prompt = data['item_prompt']
            seed = data['seed']
            prompt = data['prompt']
            if 'negative_prompt' in data:
                negative_prompt = data['negative_prompt']
            else:
                negative_prompt = ''
        else:
            llava_item_image, cat_item_image = self.read_image(data_path)
            item_prompt = 'object'
            seed = self.args.seed
            prompt = self.args.pos_prompt
            negative_prompt = self.args.neg_prompt
            
        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0

        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        # print(llava_item_tensor.shape, cat_item_tensor.shape)
        # raise ValueError
        batch = {
            "pixel_value_llava": llava_item_tensor,
            "uncond_pixel_value_llava": uncond_llava_item_tensor,
            "pixel_value_ref": cat_item_tensor,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "name": item_prompt,
            'data_name': data_name
        }
        return batch
