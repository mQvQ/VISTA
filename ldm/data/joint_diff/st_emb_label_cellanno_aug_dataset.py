from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json
import pandas as pd
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch.nn.functional as F
import pickle

class STEmbDataset(Dataset):
    def __init__(self, config):
        self.split = config.get("split")
        self.tar_cancer = config.get('tar_cancer', None)
        self.root_dir = Path(config.get("root"))
        self.emb_path = config.get('emb_path', None)
        self.prompts_file = config.get('prompts', None)
        if self.prompts_file:
            self.prompts_file = json.load(open(self.prompts_file))
        self.st_size = config.get("st_size", 256)
        self.crop_size = config.get("crop_size", None)
        self.p_uncond = config.get("p_uncond", 0)
        self.meta = self.load_meta()
        self.id_barcodes = self.prepare_dataset()
        self.use_celltype = config.get('use_celltype', False)
        self.mask_patch = config.get('mask_patch', False)
        self.use_max_cell = config.get('use_max_cell', False)
        # self.cancer_map = {'PRAD': 0, 'COAD': 1, 'READ': 2, 'CCRCC':3, 'HCC': 4, 'IHC': 5}
        self.cancer_map = {
            'prostate adenocarcinoma': 0,
            'colon adenocarcinoma': 1,
            'rectal adenocarcinoma': 2,
            'renal clear cell carcinoma': 3,
            'hepatocellular carcinoma': 4,
            'breast invasive ductal carcinoma': 5
        }
        self.organ_map = {'prostate': 0, 'bowel': 1, 'kidney lymph nodes': 2, 'liver': 3, 'axillary lymph nodes': 4}
        self.celltype_map = json.load(open(os.path.join(self.root_dir, f'celltypes_dict.json')))
        self.name = self.emb_path.split('.')[0].replace('test', 'train')
        self.mean = np.load(os.path.join(self.root_dir, f'{self.name}_mean.npy'))
        self.std = np.load(os.path.join(self.root_dir, f'{self.name}_std.npy'))
        if self.use_celltype:
            self.cell_anno = json.load(open(os.path.join(self.root_dir, f'{self.split}_set_standard_norm_celltype_newest.json')))

    def load_meta(self):
        meta = pd.read_csv(os.path.join(self.root_dir, 'meta.csv'))

        return meta

    def prepare_dataset(self):
        h5_path = os.path.join(self.root_dir, f'{self.split}_set_log1p.h5')
        with h5py.File(h5_path, 'r') as f: # sts and images: dataset
            id_barcodes = f['id_barcode'][:].tolist()
            id_barcodes = [item.decode('utf-8') for item in id_barcodes]
        self.h5_file = h5py.File(h5_path, 'r')

        case_ids = [item.split('_')[0] for item in id_barcodes]
        if self.tar_cancer:
            self.filter = [self.meta[self.meta['id'] == case_id].iloc[0]['kt_cancer_type'] == self.tar_cancer for case_id in case_ids]
            self.images = self.h5_file['img'][:][self.filter==True]
            id_barcodes = id_barcodes[self.filter==True]
            self.st_embs = pickle.load(open(os.path.join(self.root_dir, self.emb_path), 'rb'))[self.filter==True]
        else:
            self.images = self.h5_file['img']
            self.st_embs = pickle.load(open(os.path.join(self.root_dir, self.emb_path), 'rb'))

        return id_barcodes

    def process_label(self, id_barcode, idx):
        case_id, barcode = id_barcode.split('_')
        row = self.meta[self.meta['id'] == case_id].iloc[0]
        organ, cancer_type = row['kt_organ'], row['kt_cancer_type']
        label = np.array(self.cancer_map[cancer_type])
        # if self.use_celltype:
        cellannos = self.cell_anno[id_barcode] # {'cell_type': cell_types, 'cell_abundance': cell_abundance}
        final_celltypes = [self.celltype_map[0][k] for k in cellannos['cell_type']]
        cell_types_idxs = [self.celltype_map[1][k] for k in final_celltypes] 
        spot_cellanno = np.zeros(80) # 80 cell-types
        ratios = np.array(cellannos['cell_abundance'])
        ratios = ratios / np.sum(ratios)
        if self.use_max_cell:
            max_idx = np.argmax(ratios)
            spot_cellanno[cell_types_idxs[max_idx]] = 1
        else:
            for i, ratio in enumerate(ratios): spot_cellanno[cell_types_idxs[i]] = ratio

        return label, spot_cellanno


    @staticmethod
    def get_random_crop(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        img = img[y:y + size, x:x + size]
        return img

    @staticmethod
    def get_random_mask(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        n_img = img.copy()
        n_img[y:y + size, x:x + size] = 0
        return img - n_img

    def process_image(self, image):
        image = Image.fromarray(image)
        if self.mask_patch:
            image = self.get_random_mask(np.array(image), self.crop_size)
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),])
        image = transform(image)
        image = np.array(image).astype(np.float32) / 127.5 - 1
        image = torch.tensor(image.transpose((2, 0, 1)))
        return image

    def __len__(self):
        return len(self.id_barcodes)

    def process_emb(self, emb):
        emb = emb - self.mean
        emb = emb / self.std
        return emb

    def __getitem__(self, idx):
        id_barcode = self.id_barcodes[idx]
        st = self.st_embs[idx]
        st = self.process_emb(st)
        image = self.images[idx]
        # preprocess
        image = self.process_image(image)
        label, spot_cellanno = self.process_label(id_barcode, idx)
        # add unknown celltype
        spot_cellanno= np.pad(spot_cellanno, (0, 1), mode='constant', constant_values=0)

        spot_cellanno = torch.tensor(spot_cellanno)
        label = torch.tensor(label).reshape(1).to(torch.long)
        st = torch.tensor(st).reshape(1, -1) # 1, 256
        # cellanno augmentation
        if self.split == 'train':
            if np.random.rand() < 0.5:
                spot_cellanno = torch.zeros_like(spot_cellanno)
                spot_cellanno[-1] = 1
        if self.split == 'train' and np.random.rand() < self.p_uncond: # train_time; unconditional guidance
            image = torch.zeros_like(image)
            # use all 0 to represent no condition; the last celltype represents all unknown
            spot_cellanno = torch.zeros_like(spot_cellanno)
            spot_cellanno[-1] = 1
            # label = torch.tensor([-1]).reshape(1).to(torch.long)
            label = torch.tensor([-1]).reshape(1).to(torch.long)
        unconds_cellanno = torch.zeros_like(spot_cellanno)
        unconds_cellanno[-1] = 1
        # if self.use_celltype == False: # cell ratio all 0
        #     spot_cellanno = torch.zeros_like(spot_cellanno)
        return {'image': image,  'rna': st, 'conds': {'image': image, 'celltype': spot_cellanno,
                                                      'label': label},
                'unconds':{'image': torch.zeros_like(image), 'celltype': unconds_cellanno,
                           'label': torch.tensor([-1]).reshape(1).to(torch.long)}}


if __name__ == '__main__':
    config = {
        "root": "/home/tma/datasets/st-diffusion/hest_benchmark_visium_v3",
        "split": "train",
        'prompts_file': None,
        'emb_path': 'multi_organ_gae_train_set_d256_embs.pkl',
        'crop_size': 256,
        'p_uncond': 0.1,
        'sample': True,
        'use_celltype': True,
        'tar_cancer': 'prostate adenocarcinoma'
    }
    dataset = STEmbDataset(config)
    # dataloader
    print(len(dataset), len(dataset.cell_anno))
    print(dataset[0]['rna'].shape)
    import random
    for _ in range(1000):
        i = random.randint(0, len(dataset))
        rna = dataset[i]['rna']
        print(dataset[i]['conds']['label'].shape)
        print(dataset[i]['conds']['celltype'].shape)
        print(dataset[i]['conds']['image'].shape)
