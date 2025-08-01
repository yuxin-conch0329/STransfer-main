import os
import random

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
from torch.backends import cudnn

from anndata import AnnData
from sklearn import metrics

# random seed
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# preprocess
def preprocess_slice_DLPFC(slicename, data_dir, device, label_mapping, n_top_genes=1000):
    # Read spatial and metadata
    spatial_data = pd.read_csv(f"{data_dir}/{slicename}/spatial/tissue_positions_list.csv", sep=",", header=None)
    spatial_data.columns = ['barcode', 'in_tissue', 'row', 'col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    spatial_data = spatial_data[spatial_data['in_tissue'] == 1]

    meta = pd.read_csv(f"{data_dir}/{slicename}/metadata.tsv", sep="\t", index_col=False)
    meta.drop(columns=['row', 'col'], inplace=True)

    data = spatial_data.merge(meta, on='barcode', how='right')
    data = data[['barcode', 'row', 'col', 'pxl_row_in_fullres', 'pxl_col_in_fullres', 'expr_chrM']]  # expr_chrM means layer
    data = data.dropna(subset=['expr_chrM'])

    labels = data['expr_chrM'].map(label_mapping)
    labels = torch.LongTensor(labels).to(device)

    # Read and preprocess AnnData
    adata = sc.read_visium(f"{data_dir}/{slicename}")
    adata = adata[data.index]

    adata.var_names_make_unique()
    adata.layers['count'] = adata.X
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata)

    adata.obsm['process'] = adata.X

    # construct graph network
    graph_raw = graph_construction(adata, neighbor_nodes=36)
    graph_dict = {"indices": graph_raw["indices"].to(device)}

    return adata, data, labels, graph_dict

def preprocess_slice_merfish(path, label_mapping, device, n_top_genes=150):
    adata = sc.read_h5ad(path)
    labels = adata.obs['ground_truth'].map(label_mapping)
    labels = torch.LongTensor(labels).to(device)

    adata.var_names_make_unique()
    adata.layers['count'] = adata.X
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata)

    adata.obsm['process'] = adata.X
    graph_raw = graph_construction(adata, neighbor_nodes=48)
    graph_dict = {
        "indices": torch.tensor(graph_raw["indices"], dtype=torch.long).to(device)}
    return adata,labels, graph_dict

def process_slice_starmap(slicename, root_path, device, n_top_genes=140, save_h5ad=True):
    expr_path = os.path.join(root_path, f"starmap_expr_{slicename}.csv")
    spatial_path = os.path.join(root_path, f"starmap_spatial_{slicename}.csv")
    h5ad_path = os.path.join(root_path, f"starmap_{slicename}.h5ad")

    meta = pd.read_csv(expr_path, index_col=0)
    spatial_data = pd.read_csv(spatial_path, index_col=0)
    data = spatial_data.merge(meta, left_index=True, right_index=True, how='right')

    labels = data['z'].replace(4, 0)
    labels = torch.LongTensor(labels.values).to(device)

    adata = AnnData(X=meta.values)
    adata.obsm['spatial'] = spatial_data[['x', 'y']].values
    adata.var_names = meta.columns
    adata.obs_names = spatial_data.index

    if save_h5ad:
        adata.write_h5ad(h5ad_path)
        adata = sc.read_h5ad(h5ad_path)

    adata = adata[data.index]
    adata.var_names_make_unique()
    adata.layers['count'] = adata.X
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata)

    adata.obsm['process'] = adata.X
    graph_raw = graph_construction(adata, neighbor_nodes=36)
    graph_dict = {"indices": torch.tensor(graph_raw["indices"], dtype=torch.long).to(device)}
    return adata,labels, graph_dict

def process_slice_Bar(slicename, root_path, device, n_top_genes):
    slice_path = os.path.join(root_path, f'Slice_{slicename}_removed.h5ad')
    adata = sc.read_h5ad(slice_path)

    label_mapping = {'VISp_I': 0, 'VISp_II/III': 1, 'VISp_IV': 2,'VISp_V': 3, 'VISp_VI': 4, 'VISp_wm': 5}
    labels = adata.obs['layer'].map(label_mapping)
    labels = torch.LongTensor(labels).to(device)

    adata.var_names_make_unique()
    adata.layers['count'] = adata.X
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata)

    adata.obsm['process'] = adata.X

    # construct graph network
    graph_raw = graph_construction(adata, neighbor_nodes=36)
    graph_dict = {"indices": graph_raw["indices"].to(device)}

    return adata, labels, graph_dict

# constrcut graph
def graph_construction(adata, neighbor_nodes=6):
    # check ST data
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])  # 4221x4221
    n_cell = len(adata)  # number of cells
    adj_mat = np.zeros((n_cell, n_cell))
    # minimal distance nodes
    for i in range(n_cell):
        n_neighbors = np.argsort(dist[i, :])[:neighbor_nodes + 1]
        adj_mat[i, n_neighbors] = 1

    # delete itself
    x, y = np.diag_indices_from(adj_mat)
    adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)  # tranform T/F into 0/1
    adj_mat = sp.coo_matrix(adj_mat)   # store sparse matrix

    adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

    # transform into tensor
    sparse_mx = adj_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # [2, 161053]

    graph_dict = {"indices":indices}  #Construct edge indices between adjacent points
    return graph_dict

def save_model(net, filename, params):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root, filename)))


def save_predictions_to_csv(pred_labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        'spot': list(range(1, len(pred_labels) + 1)),
        'pred': pred_labels})
    df.to_csv(save_path, index=False)

def save_metrics_to_csv(acc, ari, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame([{"acc": acc,"ARI": ari}])
    df.to_csv(save_path, index=False)


