'''
ablation study on case MEND162
decode  & calc pcc.
hyper param: sample cfg.
'''
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import anndata
import scanpy as sc
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import scipy.stats as st
import json
import numpy as np

from preprocess import construct_interaction, construct_interaction_KNN, preprocess, preprocess_adj_sparse, \
    preprocess_adj
from model import GCNModelVAE_FC
from lossfunc import loss_kl, loss_zinb, loss_CE, loss_nb
from tqdm import tqdm


# calc pcc
def cal_spot_Percor(original, res, ):
    Pearson_CoPearson_Cor = pd.Series(index=gene_names)
    for i in range(res.shape[1]):
        # 要mask_idxs的
        # non_mask_idxs = np.setdiff1d(np.arange(original.shape[0]), mask_idxs)
        # calc_gt = original[non_mask_idxs, i]
        # calc_res = res[non_mask_idxs, i]
        calc_gt = original[:, i]
        calc_res = res[:, i]
        print(calc_gt.shape, calc_res.shape)
        # print(mask_idxs[0],calc_gt.shape, calc_res.shape)
        Pearson_CoPearson_Cor[i] = st.pearsonr(calc_gt, calc_res)[0]
    Pearson_Cor_mean = np.mean(Pearson_CoPearson_Cor)
    return Pearson_CoPearson_Cor, Pearson_Cor_mean


def preprocess_get_feature(adata, adata_gene_idxs, gene_counts, deconvolution=False):
    sc.pp.log1p(adata)
    # cell filter
    # cell_min_counts, cell_max_counts = 1260, 21002.5
    # sc.pp.filter_cells(adata, min_counts=cell_min_counts)
    # sc.pp.filter_cells(adata, max_counts=cell_max_counts)
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata_gene_idxs]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        X = adata_Vars.X.toarray()
    else:
        X = adata_Vars.X

    adata_Vars.obsm['feat'] = adata_Vars.X
    adata_Vars.obsm['feat_a'] = adata_Vars.X
    adata_Vars.obsm['feat_a'] = permutation(adata_Vars.obsm['feat_a'])
    adata_Vars.obsm['feat'] = adata_Vars.obsm['feat_a']
    adata_Vars.obsm['feat_a'] = adata_Vars.obsm['feat']

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

    # max-abs-scaler and add mask
    feat = feat / gene_counts

    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a


def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated


# Settings
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")  # this should be set to the GPU device you would like to use on your machine
use_cuda = True  # set to true if GPU is used

epochs = 20000  # number of training epochs
saveFreq = 200  # the model parameters will be saved during training at a frequency defined by this parameter
lr = 1e-4  # initial learning rate
weight_decay = 0  # regularization term

# v3
hidden1 = 512
hidden2 = 256
fc_dim1 = 512
dropout = 0.01  # neural network dropout term
# human_ovarian_cancer_target
XreconWeight = 10  # reconstruction weight of the gene expression
# osmFISH
# XreconWeight=1  #reconstruction weight of the gene expression
# 15

# XreconWeight=20
advWeight = 2  # weight of the adversarial loss, if used
ridgeL = 0.01  # regularization weight of the gene dropout parameter
# ridgeL=5 #regularization weight of the gene dropout parameter
# 5
training_sample_X = 'logminmax'  # specify the normalization method for the gene expression input. 'logminmax' is the default that log transforms and min-max scales the expression. 'corrected' uses the z-score normalized and ComBat corrected data from Hu et al. 'scaled' uses the same normalization as 'corrected'.
switchFreq = 10  # the number of epochs spent on training the model using one sample, before switching to the next sample
name = 'get_gae_feature'  # name of the model

datatype = '10x'

gene_names = json.load(
    open('/disk2/tma/VISTA/hest_benchmark_visium_v5/topvar_512_topmean_512_gene_names.json'))
gene_idxs = json.load(
    open('/disk2/tma/VISTA/hest_benchmark_visium_v5/topvar_512_topmean_512_genes.json'))
cell_min_counts, cell_max_counts = 1260, 21002.5
counts = np.load('/disk2/tma/VISTA/hest_benchmark_visium_v5/train_set_log1p_gene_max.npy')
gene_counts = counts[gene_idxs]

case_ids = pd.read_csv('/disk2/tma/VISTA/hest_benchmark_visium_v5/visium_test.csv')['id']
# case_ids = ['MEND162']
import h5py
from collections import defaultdict

id_barcodes = h5py.File('/disk2/tma/VISTA/hest_benchmark_visium_v5/test_set_log1p.h5')['id_barcode']
id_barcodes = [item.decode('utf-8') for item in id_barcodes]
id2barcodes = defaultdict(list)
for item in id_barcodes:
    i, barcode = item.split('_')
    id2barcodes[i].append(barcode)

all_adatas, all_feats, all_adj_norms, all_pos_weights, all_norms, all_adjs = [], [], [], [], [], []

for i, case_id in enumerate(case_ids):
    adata = sc.read(os.path.join('/disk2/tma/VISTA/r10-hest-st', case_id + '.h5ad'))
    barcodes = id2barcodes[case_id]
    adata = adata[barcodes, :]
    print(adata.shape)
    # gen latent embeddings
    if 'feat' not in adata.obsm.keys():
        adata_genes = adata.var['genome'].index.tolist()
        adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
        # spot imputation
        preprocess_get_feature(adata, adata_gene_idxs=adata_gene_idxs, gene_counts=gene_counts)

    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:
            construct_interaction_KNN(adata)
    features = torch.tensor(adata.obsm['feat'].copy()).to(device)
    adj = adata.obsm['adj']
    graph_neigh = torch.tensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(device)

    dim_input = features.shape[1]
    dim_output = hidden2

    if datatype in ['Stereo', 'Slide']:
        # using sparse
        print('Building sparse matrix ...')
        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj + sp.eye(adj.shape[0])).to(device)
    else:

        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj + sp.eye(adj.shape[0])).to(device)
    all_feats.append(features)
    all_adj_norms.append(adj_norm)
    all_adjs.append(adj)
    all_adatas.append(adata)
    all_pos_weights.append(pos_weight)
    all_norms.append(norm)

num_features = all_feats[0].shape[1]
model = GCNModelVAE_FC(num_features, hidden1, hidden2, fc_dim1, dropout).to(device)
model.load_state_dict(
    torch.load('/disk2/tma/VISTA/hest_benchmark_visium_v5_trainset_gae_d256_3k_epoch.ckpt'))
model.to(device).eval()
import glob

# search_path = '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-infer-times-ablation/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/'
# search_path = '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/'
search_path = '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
# preds_files = [
#     # '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-02_11-38-16-test-infer-0-cfg-0.0-idx-58944.npy',
#     # '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-02_13-30-05-test-infer-0-cfg-1.0-idx-58944.npy',
#     # '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/mixcontrol-single-gen-st-embs-wo-celltype-0-cfg-4.5.npy',
#     # '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-01_21-29-15-test-infer-0-cfg-20.0-idx-3072.npy'
#     '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-05_00-19-44-test-infer-0-cfg-20.0-idx-58944.npy'
# ]
preds_files = [
    # '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-0.0.npy',
    # '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-1.0.npy',
    # '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-4.5.npy',
    '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-20.0.npy',
]
cfgs = [20]
# cfgs = [20]
emb_means = np.load(
    '/disk2/tma/VISTA/hest_benchmark_visium_v5/multi_organ_gae_train_set_log1p_d256_embs_3+1_mean.npy')
emb_stds = np.load(
    '/disk2/tma/VISTA/hest_benchmark_visium_v5/multi_organ_gae_train_set_log1p_d256_embs_3+1_std.npy')
pred_latents = []
for pred_file in preds_files:
    preds = np.load(pred_file)
    preds = preds * emb_stds + emb_means
    pred_latents.append(preds)

idx = 0
for i in range(len(all_feats)):
    all_features_recon, all_z = [], []
    all_pccs = []
    all_preds = []
    # cfgs = [0, 1, 4.5, 20]
    # cfgs = [1, 4.5]
    for j in range(len(pred_latents)):
        features = all_feats[i]
        adj_norm = all_adj_norms[i]
        adata = all_adatas[i]
        latent = pred_latents[j][idx:idx + len(features)]
        latent = torch.tensor(latent, dtype=torch.float32).to(device).squeeze(1)
        pred_recon = model.decoder(latent, adj_norm)
        recon = pred_recon[3].detach().cpu().numpy()
        final = np.clip(recon, 0, 1)
        final = final * gene_counts
        pcc, pcc_mean = cal_spot_Percor(features.detach().cpu().numpy() * gene_counts, final)
        all_pccs.append(pcc)
        adata = all_adatas[i].copy()
        adata.obsm['pred'] = final
        adata.obsm['gt'] = features.detach().cpu().numpy() * gene_counts
        all_preds.append(adata)
        # top50_means = [all_pccs[j].nlargest(50).mean() for i in range(len(all_pccs))]
        # overall_means = [all_pccs[j].mean() for i in range(len(all_pccs))]
        # if case_ids[i] == 'ZEN45':
        if case_ids[i] == 'MEND162':
            # vis_save_dir = f'./ablation-cfg-vis-zen45/cfgs_{cfgs[j]}'
            vis_save_dir = f'./ablation-cfg-vis-mend162/cfgs_{cfgs[j]}'
            os.makedirs(vis_save_dir, exist_ok=True)
            # gene_names = all_pccs[j].nlargest(20).index.tolist()
            # gene_names= ['KLK3', 'NKX3-1'] # prad
            # gene_names = ['AEBP1']
            gene_names = ['KLK3']
            gene_to_position = {gene: i for i, gene in enumerate(pcc.index)}
            for name in tqdm(gene_names):
                plt.rcParams["figure.figsize"] = (8, 8)
                adata.obs[f'{name}_pred_counts'] = adata.obs['log1p_total_counts'].copy()
                adata.obs[f'{name}_pred_counts'] = adata.obsm['pred'][:, gene_to_position[name]]
                sc.pl.spatial(adata, img_key='downscaled_fullres', color=[f'{name}_pred_counts'], size=1.75)
                # save image
                plt.savefig(os.path.join(vis_save_dir, f'{case_ids[i]}_{name}_pred_{all_pccs[j][name]}.png'))
                plt.close()
                # save gt
                plt.rcParams["figure.figsize"] = (8, 8)
                adata.obs[f'{name}_gt_counts'] = adata.obs['log1p_total_counts'].copy()
                adata.obs[f'{name}_gt_counts'] = adata.obsm['gt'][:, gene_to_position[name]]
                sc.pl.spatial(adata, img_key='downscaled_fullres', color=[f'{name}_gt_counts'], size=1.75)
                plt.savefig(os.path.join(vis_save_dir, f'{case_ids[i]}_{name}_gt.png'))
                plt.close()
    idx += len(all_feats[i])







