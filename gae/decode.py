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


def preprocess_get_feature(adata, adata_gene_idxs, gene_counts, deconvolution=False):
    print(adata)
    sc.pp.log1p(adata)
    print(adata)
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata_gene_idxs]

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
    if 'feat' not in adata.obsm.keys():
        adata_genes = adata.var['genome'].index.tolist()
        adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
        print(adata[:, adata_gene_idxs])
        preprocess_get_feature(adata, adata_gene_idxs=adata_gene_idxs, gene_counts=gene_counts)

    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:
            # construct_interaction(adata)
            construct_interaction_KNN(adata)
    features = torch.tensor(adata.obsm['feat'].copy()).to(device)
    adj = adata.obsm['adj']
    graph_neigh = torch.tensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(device)

    dim_input = features.shape[1]
    dim_output = hidden2

    # get adj
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

mean = np.load(
    '/disk2/tma/VISTA/hest_benchmark_visium_v5/multi_organ_gae_train_set_log1p_d256_embs_3+1_mean.npy')
std = np.load(
    '/disk2/tma/VISTA/hest_benchmark_visium_v5/multi_organ_gae_train_set_log1p_d256_embs_3+1_std.npy')

import numpy as np

# results = np.load('/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-celltype-infer/02-24T18-53_single_st_unet_mixcontrolv3_hv5/mixcontrol-single-gen-st-embs-wo-celltype.npy')
# save_dir = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-celltype-infer/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
# pred_path = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-image-infer/02-24T18-53_single_st_unet_mixcontrolv3_hv5/mixcontrol-single-gen-st-embs-wo-img.npy'
# pred_path = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-cancer-infer/02-24T18-53_single_st_unet_mixcontrolv3_hv5/mixcontrol-single-gen-st-embs-wo-cancer.npy'
# pred_path = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-cancer-infer-cfg-1/02-24T18-53_single_st_unet_mixcontrolv3_hv5/mixcontrol-single-gen-st-embs-wo-cancer.npy'
# results = np.load('/home/tma/ST-Diffusion/results/st-single-generation-emb-test/01-19T05-40_single_st_unet_imgcond_hest_benchmark_visium_v5_gae_emb/single-gen-st-embs.npy')

# pred_path = '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-02_13-30-05-test-infer-0-cfg-1.0-idx-58944.npy'
#
# # save_dir = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-wo-image-infer/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
# save_dir = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-1/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
# pred_path = '/data1/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-ablation-all-cases/02-24T18-53_single_st_unet_mixcontrolv3_hv5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-02_11-38-16-test-infer-0-cfg-0.0-idx-58944.npy'
# save_dir = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-wo-celltype-pred-cfg-0/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
#
# pred_path = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-random-cancer-infer-cfg-4.5/02-24T18-53_single_st_unet_mixcontrolv3_hv5/2025-06-09 12:48:51-mixcontrol-single-gen-st-embs-random-cancer.npy'
# save_dir = '/home/tma/ST-Diffusion/results_hv5/mixcontrol-train-random-cancer-infer-cfg-4.5/02-24T18-53_single_st_unet_mixcontrolv3_hv5'
pred_path = '/disk2/tma/VISTA/logs/10-08T20-45_st_unet/results/10-08T20-45_st_unet/mixcontrol-st-embs-allcond-test-4.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-08T20-45_st_unet/results/mixcontrol-st-embs-allcond-test-4.5'
pred_path = '/disk2/tma/VISTA/logs/10-09T06-08_ft_st_unet/results/10-08T20-45_st_unet/mixcontrol-st-embs-wo-celltype-test-3.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-09T06-08_ft_st_unet/results/mixcontrol-st-embs-wo-celltype-test-3.0'
pred_path = '/disk2/tma/VISTA/logs/10-08T20-45_st_unet/results/10-08T20-45_st_unet/mixcontrol-st-embs-wo-celltype-test-3.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-08T20-45_st_unet/results/mixcontrol-st-embs-wo-celltype-test-3.0'
pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-4.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-4.5'
pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-3.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-3'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-allcond-test-4.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-allcond-test-4.5'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-img-test-wo-celltype-4.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-img-test-wo-celltype-4.5'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-img-test-4.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-img-test-4.5'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-0.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-0.0'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-1.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-1.0'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-1.5.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-1.5'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-6.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-6.0'

pred_path = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/10-10T17-05_st_unet_mixcontrolv6/mixcontrol-st-embs-wo-celltype-test-20.0.npy'
save_dir = '/disk2/tma/VISTA/logs/10-10T17-05_st_unet_mixcontrolv6/results/mixcontrol-st-embs-wo-celltype-test-20.0'

os.makedirs(save_dir, exist_ok=True)
results = np.load(pred_path)
results = results * std + mean

model.eval()
count = 0
pred_recons = []
with torch.no_grad():
    for i in range(len(all_feats)):
        features = all_feats[i]
        adj_norm = all_adj_norms[i]
        case_id = case_ids[i]
        z = torch.tensor(results[count: count + features.shape[0]]).to(device).squeeze(1)
        count += features.shape[0]
        # decode z
        pred_recon = model.decoder(z, adj_norm)[3].cpu().numpy() * gene_counts
        # save
        print(pred_recon.shape)
        np.save(os.path.join(save_dir, f'pred_{case_id}.npy'), pred_recon)

# save
