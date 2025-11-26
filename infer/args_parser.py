# arg_parser.py
import argparse

parser = argparse.ArgumentParser(description='Convert variables to argparse.')

parser.add_argument('--sample_num', type=int, default=10, help='SAMPLE_NUM')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--split', type=str, default='test', help='')
parser.add_argument('--sample_by_cls', action='store_true', help='whether Sample by class')
parser.add_argument('--tar_organ', type=str, default=None, help='Target organ')
parser.add_argument('--tar_cancer_type', type=str, default=None, help='Target cancer type')
parser.add_argument('--img_shape', nargs=3, type=int, default=[3, 64, 64], help='Image shape')
parser.add_argument('--st_shape', nargs=2, type=int, default=[1, 1024], help='ST shape')
parser.add_argument('--exp_name', type=str, help='Experiment name')
parser.add_argument('--scale', type=float, default=3.0, help='Scale of classifier free guidance')
parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
parser.add_argument('--config_path', type=str, help='Config path')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for torch')
parser.add_argument('--results_save_dir', type=str, help='Results save directory')
parser.add_argument('--emb_path', type=str)
parser.add_argument('--mask_ratio', type=float, default=0.2, help='Mask ratio')
parser.add_argument('--nouncond', type=bool, default=False) # for ft imputation
parser.add_argument('--celltype', type=bool, default=False) # dataset caption是否使用celltype
parser.add_argument('--root', type=str, help='data root dir')
parser.add_argument('--case_id', type=str, default=None, help='single slide dataset case id')
parser.add_argument('--celltype_index', type=int, default=None, help='perturbe cell type')
parser.add_argument('--celltype_name', type=str, default=None, help='perturbe cell type name')