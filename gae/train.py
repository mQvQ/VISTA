'''
hest_benchmark_visium_v5 train
没有cell-norm
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

from preprocess import construct_interaction, construct_interaction_KNN,preprocess, preprocess_adj_sparse, preprocess_adj
from model import GCNModelVAE_FC
from lossfunc import loss_kl,loss_zinb,loss_CE,loss_nb

# Settings
device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")#this should be set to the GPU device you would like to use on your machine
use_cuda=True #set to true if GPU is used
# seed=3 #random seed
# testNodes=0.1 #fraction of total cells used for testing
# valNodes=0.05 #fraction of total cells used for validation
# useSavedMaskedEdges=True #some edges of the adjacency matrices are held-out for validation; set to True to save and use saved version of the edge masks
####1600########
neighbor_num=3
epochs= 10000 #number of training epochs
####1600########
saveFreq=2000 #the model parameters will be saved during training at a frequency defined by this parameter
lr=1e-4 #initial learning rate
weight_decay=0 #regularization term

# hidden1=256 #Number of units in hidden layer 1
# hidden2=64 #Number of units in hidden layer 2
# fc_dim1=256 #Number of units in the fully connected layer of the decoder
# v3
hidden1 = 512
hidden2 = 256
fc_dim1 = 512
dropout=0.01 #neural network dropout term
#human_ovarian_cancer_target
XreconWeight=10  #reconstruction weight of the gene expression
#osmFISH
# XreconWeight=1  #reconstruction weight of the gene expression
#15
# XreconWeight=20
advWeight=2 # weight of the adversarial loss, if used
ridgeL=0.01 #regularization weight of the gene dropout parameter
# ridgeL=5 #regularization weight of the gene dropout parameter
#5
training_sample_X='logminmax' #specify the normalization method for the gene expression input. 'logminmax' is the default that log transforms and min-max scales the expression. 'corrected' uses the z-score normalized and ComBat corrected data from Hu et al. 'scaled' uses the same normalization as 'corrected'.
switchFreq=10 #the number of epochs spent on training the model using one sample, before switching to the next sample
name='get_gae_feature' #name of the modelq
datatype='10x'

gene_names = json.load(open('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v4/topvar_512_topmean_512_gene_names.json'))
gene_idxs = json.load(open('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v4/topvar_512_topmean_512_genes.json'))
counts = np.load('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/train_set_log1p_gene_max.npy')
gene_counts = counts[gene_idxs]

meta = pd.read_csv('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v4/visium_train.csv')
adata = sc.read(os.path.join('/data1/tma/HEST/hest_data/st', meta['id'][0] + '.h5ad'))


def preprocess_get_feature(adata, adata_gene_idxs, gene_counts, deconvolution=False):
    # sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata_gene_idxs]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

        # max-abs-scaler
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

import h5py
from collections import defaultdict
id_barcodes = h5py.File('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/train_set_log1p.h5')['id_barcode']
id_barcodes = [item.decode('utf-8') for item in id_barcodes]
id2barcodes = defaultdict(list)
for item in id_barcodes:
    i, barcode = item.split('_')
    id2barcodes[i].append(barcode)

all_adatas, all_feats, all_adj_norms, all_pos_weights, all_norms, all_adjs = [], [], [], [], [], []

for case_id in meta['id']:
    adata = sc.read(os.path.join('/data1/tma/HEST/hest_data/st', case_id + '.h5ad'))
    barcodes = id2barcodes[case_id]
    adata = adata[barcodes, :]
    if 'feat' not in adata.obsm.keys():
        adata_genes = adata.var['genome'].index.tolist()
        adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
        preprocess_get_feature(adata, adata_gene_idxs=adata_gene_idxs, gene_counts=gene_counts)

    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata, n_neighbors=neighor_num)
        else:
          # construct_interaction(adata)
            construct_interaction_KNN(adata, n_neighbors=neighbor_num)
    features = torch.tensor(adata.obsm['feat'].copy()).to(device)
    adj = adata.obsm['adj']
    graph_neigh = torch.tensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(device)

    dim_input = features.shape[1]
    dim_output = hidden2

    #get adj
    if datatype in ['Stereo', 'Slide']:
        #using sparse
        print('Building sparse matrix ...')
        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj+ sp.eye(adj.shape[0])).to(device)
    else:
        # standard version
        # adj_norm = preprocess_adj(adj)
        # adj_norm = torch.FloatTensor(adj_norm).to(device)
        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj+ sp.eye(adj.shape[0])).to(device)
    all_feats.append(features)
    all_adj_norms.append(adj_norm)
    all_adjs.append(adj)
    all_adatas.append(adata)
    all_pos_weights.append(pos_weight)
    all_norms.append(norm)

def cal_Percor(original,res):
    Pearson_CoPearson_Cor = pd.Series(index=original.var_names)
    for i in range(res.X.shape[1]):
        Pearson_CoPearson_Cor[i]=st.pearsonr(original.X[:, i],res.X[:,i])[0]
    Pearson_Cor_mean = np.mean(Pearson_CoPearson_Cor)
    return Pearson_CoPearson_Cor,Pearson_Cor_mean

num_features = all_feats[0].shape[1]
model = GCNModelVAE_FC(num_features, hidden1,hidden2,fc_dim1, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(epochs):
    train_loss_ep = [None] * epochs
    train_loss_kl_ep = [None] * epochs
    train_loss_x_ep = [None] * epochs
    train_loss_a_ep = [None] * epochs
    for epoch in range(epochs):
        # maskedgeres= mask_nodes_edges(features.shape[0],testNodeSize=testNodes,valNodeSize=valNodes,seed=seed)
        # train_nodes_idx,val_nodes_idx,test_nodes_idx = maskedgeres
        t = time.time()
        model.train()
        for i in range(len(all_feats)):
            features = all_feats[i]
            adj_norm = all_adj_norms[i]
            adata = all_adatas[i]
            pos_weight = all_pos_weights[i]
            norm = all_norms[i]
            adj = all_adjs[i]
            adata_genes = adata.var['genome'].index.tolist()
            adata_gene_idxs = [adata_genes.index(x) for x in gene_names]

            optimizer.zero_grad()

            adj_recon, mu, logvar, z, features_recon = model(features, adj_norm)
            loss_kl_train = loss_kl(mu, logvar)
            loss_x_train = loss_zinb(features_recon, features, XreconWeight, ridgeL)
            loss_function = nn.MSELoss()
            loss_r_train = loss_function(features_recon[3], features)
            # loss_x_train = loss_nb(features_recon, features,XreconWeight)
            loss_a_train = loss_CE(adj_recon, adj, pos_weight, norm)

            loss = loss_kl_train + loss_x_train + loss_r_train
            # loss=loss_x_train+loss_r_train
            # loss = loss_kl_train+loss_x_train+loss_a_train
            loss.backward()
            optimizer.step()
            train_loss_ep[epoch], train_loss_kl_ep[epoch], train_loss_x_ep[epoch], train_loss_a_ep[
                epoch] = loss.item(), loss_kl_train.item(), loss_x_train.item(), loss_a_train.item()
            if epoch % saveFreq == 0:
                print(' Epoch: {:04d}'.format(epoch),
                      'loss_train: {:.4f}'.format(loss.item()),
                      'loss_kl_train: {:.4f}'.format(loss_kl_train.item()),
                      'loss_x_train: {:.4f}'.format(loss_x_train.item()),
                      'loss_a_train: {:.4f}'.format(loss_a_train.item()),
                      'time: {:.4f}s'.format(time.time() - t))

                sam = adata.obsm['feat']
                com = features_recon[3].detach().cpu().numpy()
                sam = anndata.AnnData(sam, var=adata[:, adata_gene_idxs].var)
                com = anndata.AnnData(com)
                our_Percor, our_Percor_mean = cal_Percor(sam, com)

                print(our_Percor_mean)

    model.to(device).eval()
    all_features_recon, all_z = [], []
    for i in range(len(all_feats)):
        features = all_feats[i]
        adj_norm = all_adj_norms[i]
        adata = all_adatas[i]

        adj_recon, mu, logvar, z, features_recon = model(features, adj_norm)

        all_features_recon.append(features_recon)
        all_z.append(z)

    return train_loss_ep, train_loss_kl_ep, train_loss_x_ep, train_loss_a_ep, all_z, all_features_recon

t_ep=time.time()
train_loss_ep,train_loss_kl_ep,train_loss_x_ep,train_loss_a_ep, all_z, all_features_recon=train(epochs=epochs)
print(' total time: {:.4f}s'.format(time.time() - t_ep))

all_pccs = []
for i in range(len(all_feats)):
    adata = all_adatas[i]
    recon = all_features_recon[i]
    sam = adata.obsm['feat']
    com = all_features_recon[i][3].detach().cpu().numpy()
    adata_genes = adata.var['genome'].index.tolist()
    adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
    sam = anndata.AnnData(sam,var = adata[:,adata_gene_idxs].var)
    com = anndata.AnnData(com)
    def cal_Percor(original,res):
        Pearson_CoPearson_Cor = pd.Series(index=original.var_names)
        for i in range(res.X.shape[1]):
            Pearson_CoPearson_Cor[i]=st.pearsonr(original.X[:, i],res.X[:, i])[0]
        Pearson_Cor_mean = np.mean(Pearson_CoPearson_Cor)
        return Pearson_CoPearson_Cor,Pearson_Cor_mean
    our_Percor,our_Percor_mean = cal_Percor(sam,com)
    all_pccs.append(our_Percor_mean)
print(np.mean(all_pccs))

name = 'get_gae_feature'
modelsavepath='./'+name
os.makedirs(modelsavepath, exist_ok=True)
torch.save(model.cpu().state_dict(), os.path.join(modelsavepath, f'reproduce_{epochs}.ckpt'))

# extract embs for train and test

import pickle

model.to(device).eval()
zs = []
with torch.no_grad():
    for i in range(len(all_feats)):
        case_id = meta['id'][i]
        features = all_feats[i]
        adj_norm = all_adj_norms[i]
        adata = all_adatas[i]
        pos_weight = all_pos_weights[i]
        norm = all_norms[i]
        adj = all_adjs[i]
        adata_genes = adata.var['genome'].index.tolist()
        adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
        adj_recon, mu, logvar, z, features_recon = model(features, adj_norm)
        # save z
        z = z.cpu().numpy()
        zs.append(z)
    zs = np.concatenate(zs, axis=0)
    with open(f'/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/multi_organ_gae_train_set_d256_embs_{neighbor_num}+1.pkl', 'wb') as file:
        pickle.dump(zs, file)

# ext test set feats
test_meta = pd.read_csv('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/visium_test.csv')
all_adatas, all_feats, all_adj_norms, all_pos_weights, all_norms, all_adjs = [], [], [], [], [], []
id_barcodes = h5py.File('/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/test_set_log1p.h5')['id_barcode']
id_barcodes = [item.decode('utf-8') for item in id_barcodes]
id2barcodes = defaultdict(list)
for item in id_barcodes:
    i, barcode = item.split('_')
    id2barcodes[i].append(barcode)
for case_id in test_meta['id']:
    adata = sc.read(os.path.join('/data1/tma/HEST/hest_data/st', case_id + '.h5ad'))

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

    #get adj
    if datatype in ['Stereo', 'Slide']:
        #using sparse
        print('Building sparse matrix ...')
        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj+ sp.eye(adj.shape[0])).to(device)
    else:
        # standard version
        # adj_norm = preprocess_adj(adj)
        # adj_norm = torch.FloatTensor(adj_norm).to(device)
        adj_norm = preprocess_adj_sparse(adj).to(device)
        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        adj = torch.tensor(adj+ sp.eye(adj.shape[0])).to(device)
    all_feats.append(features)
    all_adj_norms.append(adj_norm)
    all_adjs.append(adj)
    all_adatas.append(adata)
    all_pos_weights.append(pos_weight)
    all_norms.append(norm)


# ext test set feats
import pickle

model.eval()
zs = []
with torch.no_grad():
    for i in range(len(all_feats)):
        case_id = test_meta['id'][i]
        features = all_feats[i]
        adj_norm = all_adj_norms[i]
        adata = all_adatas[i]
        pos_weight = all_pos_weights[i]
        norm = all_norms[i]
        adj = all_adjs[i]
        adata_genes = adata.var['genome'].index.tolist()
        adata_gene_idxs = [adata_genes.index(x) for x in gene_names]
        adj_recon,mu,logvar,z,features_recon = model(features, adj_norm)
        # save z
        z = z.cpu().numpy()
        zs.append(z)
        
    zs = np.concatenate(zs, axis=0)
    with open(f'/home/tma/datasets/st-diffusion/hest_benchmark_visium_v5/multi_organ_gae_test_set_d256_embs_{neighbor_num}+1.pkl', 'wb') as file:
        pickle.dump(zs, file)


