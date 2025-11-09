#!/usr/bin/env python3
"""
experiment2_v3.py

Compare PCA (50 PCs) vs scVI latent embeddings (50 dims) on the CellBench
dataset. For each removed label the script:
 - constructs training (annotated) and test (unannotated) splits as in
   `experiment2_v2.py`,
 - builds embeddings using PCA (50) and, if scvi-tools is available, scVI
   (latent_dim=50),
 - for each embedding type (PCA50, scVI) runs MARS, SRNC and Rejection (timed),
 - collects metrics (ARI, Accuracy, Recall, Precision, F1) and timings,
 - writes CSVs and plots to `./results/experiment2/scvi_vs_pca50/`.

Notes:
 - scvi-tools is optional. If it's not installed the script will still run
   PCA50 and produce comparative output for PCA only.
 - scVI training can be slow; default epochs are taken from the project's
   CLI parser (if available) or 50 epochs. Reduce epochs for quick tests.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, precision_score, recall_score

from model.mars.mars import MARS
from model.mars.experiment_dataset import ExperimentDataset
from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2
from model.args_parser import get_parser

# Settings
pc_n = 50
latent_dim_scvi = 50
data_h5ad = './data/bench/cellbench.h5ad'
out_dir = './results/experiment2/compare_scvi_pca'
os.makedirs(out_dir, exist_ok=True)

predictive_alg = 'lightGBM'
control_neighbor = 5
threshold_rejection = 0.7
filter_proportion = 5
shrink_parameter = 1

# parse args to reuse project defaults (epochs, cuda flags, etc.)
params, unknown = get_parser().parse_known_args()

# device selection
if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available():
    device = 'cuda:0'
    print('[INFO] Using CUDA GPU')
elif torch.backends.mps.is_available():
    device = 'mps'
    print('[INFO] Using MPS')
else:
    device = 'cpu'
    print('[INFO] Using CPU')
params.device = device

# helper metrics
def safe_metrics(y_true, y_pred):
    return {
        'adj_rand': float(adjusted_rand_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }

# load data (Anndata H5AD)
import scanpy as sc
print('Loading', data_h5ad)
adata_raw = sc.read_h5ad(data_h5ad)

# keep counts if present
try:
    adata_raw.layers['counts'] = adata_raw.X.copy()
except Exception:
    pass

batch_key = 'experiment'
label_key = 'ground_truth'

# work on highly-variable genes as in experiment2_v2
sc.pp.highly_variable_genes(adata_raw, n_top_genes=2000, flavor='cell_ranger', batch_key=batch_key)
adata_hvg = adata_raw[:, adata_raw.var['highly_variable']].copy()

# compute PCA(n_components=pc_n)
print(f'Computing PCA (n_components={pc_n})')
sc.pp.pca(adata_hvg, n_comps=pc_n)
X_pca = adata_hvg.obsm['X_pca']
labels_all = adata_hvg.obs[label_key].values
experiments_all = adata_hvg.obs[batch_key].values

# build scVI latent if scvi is available
scvi_available = True
try:
    import scvi
    from scvi.model import SCVI
except Exception:
    scvi_available = False
    print('scvi-tools not available; scVI embedding will be skipped.')

if scvi_available:
    try:
        print('Setting up scVI (this may take a while)')
        SCVI = scvi.model.SCVI
        scvi.model.SCVI.setup_anndata(adata_hvg)
        n_epochs = getattr(params, 'epochs', 50) or 50
        scvi_model = SCVI(adata_hvg, latent_distribution='normal', n_latent=latent_dim_scvi)
        scvi_model.train(max_epochs=int(n_epochs))
        latent = scvi_model.get_latent_representation()
    except Exception as e:
        print('scVI training failed, skipping scVI embedding:', e)
        scvi_available = False

# choose two experiments as in experiment2_v2 (10x_5cl vs CelSeq2_5cl)
mask_10x = experiments_all == '10x_5cl'
mask_celseq2 = experiments_all == 'CelSeq2_5cl'

X_10x_pca = X_pca[mask_10x]
y_10x = labels_all[mask_10x]
cells_10x = np.array(adata_hvg.obs_names)[mask_10x]

X_celseq2_pca = X_pca[mask_celseq2]
y_celseq2 = labels_all[mask_celseq2]
cells_celseq2 = np.array(adata_hvg.obs_names)[mask_celseq2]

if scvi_available:
    latent_full = latent
    X_10x_scvi = latent_full[mask_10x]
    X_celseq2_scvi = latent_full[mask_celseq2]

# containers
metrics_rows = []
timings_rows = []

removed_labels_unique = np.unique(y_10x)
print('Removed labels to iterate:', removed_labels_unique)

for remove_label in removed_labels_unique:
    print('\nRemoving label:', remove_label)
    train_mask = y_10x != remove_label
    if np.sum(train_mask) == 0:
        print('Skipping label (no training examples left)')
        continue

    # prepare training and test sets for PCA50
    X_train_pca = X_10x_pca[train_mask]
    y_train_pca = y_10x[train_mask]
    X_test_pca = X_celseq2_pca
    y_test_pca = y_celseq2

    # prepare training and test sets for scVI (if available)
    if scvi_available:
        X_train_scvi = X_10x_scvi[train_mask]
        y_train_scvi = y_10x[train_mask]
        X_test_scvi = X_celseq2_scvi
        y_test_scvi = y_celseq2

    # iterate over embedding types
    for embedding_name in (['PCA50'] + (['scVI'] if scvi_available else [])):
        print('Processing embedding:', embedding_name)
        if embedding_name == 'PCA50':
            annotated_set = ExperimentDataset(np.asarray(X_train_pca), np.asarray(cells_10x[train_mask]), np.asarray([f'PC{i+1}' for i in range(pc_n)]), '10x', np.asarray(y_train_pca))
            unannotated_set = ExperimentDataset(np.asarray(X_test_pca), np.asarray(cells_celseq2), np.asarray([f'PC{i+1}' for i in range(pc_n)]), 'celseq2', np.asarray(y_test_pca))
            pretrain_set = ExperimentDataset(np.asarray(X_test_pca), np.asarray(cells_celseq2), np.asarray([f'PC{i+1}' for i in range(pc_n)]), 'celseq2')
        else:
            annotated_set = ExperimentDataset(np.asarray(X_train_scvi), np.asarray(cells_10x[train_mask]), np.asarray([f'LAT{i+1}' for i in range(latent_dim_scvi)]), '10x', np.asarray(y_train_scvi))
            unannotated_set = ExperimentDataset(np.asarray(X_test_scvi), np.asarray(cells_celseq2), np.asarray([f'LAT{i+1}' for i in range(latent_dim_scvi)]), 'celseq2', np.asarray(y_test_scvi))
            pretrain_set = ExperimentDataset(np.asarray(X_test_scvi), np.asarray(cells_celseq2), np.asarray([f'LAT{i+1}' for i in range(latent_dim_scvi)]), 'celseq2')

        # MARS
        mars_time = None
        mars_metrics = {}
        try:
            t0 = time.perf_counter()
            mars = MARS(len(np.unique(unannotated_set.y)), params, [annotated_set], unannotated_set, pretrain_set, hid_dim_1=1000, hid_dim_2=100)
            out = mars.train(evaluation_mode=True, save_all_embeddings=False)
            t1 = time.perf_counter()
            mars_time = t1 - t0
            scores = None
            if isinstance(out, tuple):
                if len(out) == 3:
                    _, _, scores = out
                elif len(out) == 2:
                    _, scores = out
            else:
                scores = out
            if isinstance(scores, dict):
                for k in ['adj_rand','accuracy','recall','precision','f1_score']:
                    if k in scores:
                        try:
                            mars_metrics[k] = float(scores[k])
                        except Exception:
                            mars_metrics[k] = None
        except Exception as e:
            print('MARS error:', e)
            mars_time = None
            mars_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        # SRNC
        try:
            t0 = time.perf_counter()
            # build combined data used by SRNC/rejection
            if embedding_name == 'PCA50':
                data_embbed_x = np.concatenate([np.asarray(X_train_pca), np.asarray(X_test_pca)])
                data_embbed_y = np.concatenate([np.asarray(y_train_pca), np.asarray(y_test_pca)])
                annotated_x = np.asarray(X_train_pca)
                unannotated_x = np.asarray(X_test_pca)
                annotated_y = np.asarray(y_train_pca)
            else:
                data_embbed_x = np.concatenate([np.asarray(X_train_scvi), np.asarray(X_test_scvi)])
                data_embbed_y = np.concatenate([np.asarray(y_train_scvi), np.asarray(y_test_scvi)])
                annotated_x = np.asarray(X_train_scvi)
                unannotated_x = np.asarray(X_test_scvi)
                annotated_y = np.asarray(y_train_scvi)

            y_all_labels = list(set(data_embbed_y))
            Y_pred_srnc = SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg, control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)
            srnc_time = time.perf_counter() - t0
        except Exception as e:
            print('SRNC error:', e)
            Y_pred_srnc = None
            srnc_time = None

        # Rejection
        try:
            t0 = time.perf_counter()
            Y_pred_rej = classification_rejection_v2(data_embbed_x, data_embbed_y, list(set(data_embbed_y)), annotated_x, annotated_y, unannotated_x, predictive_alg, threshold_rejection)
            rej_time = time.perf_counter() - t0
        except Exception as e:
            print('Rejection error:', e)
            Y_pred_rej = None
            rej_time = None

        # metrics
        if Y_pred_srnc is not None:
            srnc_metrics = safe_metrics(unannotated_set.y, Y_pred_srnc)
        else:
            srnc_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        if Y_pred_rej is not None:
            rej_metrics = safe_metrics(unannotated_set.y, Y_pred_rej)
        else:
            rej_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        metrics_rows.append({
            'embedding': embedding_name,
            'removed_label': int(remove_label),
            'method': 'MARS',
            **{k: mars_metrics.get(k, None) for k in ['adj_rand','accuracy','recall','precision','f1_score']}
        })
        metrics_rows.append({
            'embedding': embedding_name,
            'removed_label': int(remove_label),
            'method': 'SRNC',
            **srnc_metrics
        })
        metrics_rows.append({
            'embedding': embedding_name,
            'removed_label': int(remove_label),
            'method': 'Rejection',
            **rej_metrics
        })

        timings_rows.append({
            'embedding': embedding_name,
            'removed_label': int(remove_label),
            'mars_time_s': mars_time,
            'srnc_time_s': srnc_time,
            'rejection_time_s': rej_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        })

# save results
metrics_df = pd.DataFrame(metrics_rows)
timings_df = pd.DataFrame(timings_rows)
metrics_csv = os.path.join(out_dir, 'metrics_scvi_vs_pca50.csv')
timings_csv = os.path.join(out_dir, 'timings_scvi_vs_pca50.csv')
metrics_df.to_csv(metrics_csv, index=False)
timings_df.to_csv(timings_csv, index=False)
print('Saved metrics ->', metrics_csv)
print('Saved timings ->', timings_csv)

# plotting
try:
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    metric_label_map = {'adj_rand': 'ARI', 'accuracy': 'Accuracy', 'recall': 'Recall', 'precision': 'Precision', 'f1_score': 'F1.score'}
    for metric in ['adj_rand','accuracy','recall','precision','f1_score']:
        display_name = metric_label_map.get(metric, metric)
        plt.figure(figsize=(6,4))
        sns.lineplot(data=metrics_df, x='removed_label', y=metric, hue='embedding', style='method', estimator='mean', ci='sd', marker='o')
        plt.title(f'{display_name}')
        plt.xlabel('removed_label')
        plt.ylabel(display_name)
        plt.legend(title='', loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{metric}_by_removed_label.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # timing: compare total run time per embedding
    if not timings_df.empty:
        tdf = timings_df.copy()
        tdf['total_time'] = tdf[['mars_time_s','srnc_time_s','rejection_time_s']].sum(axis=1, skipna=True)
        plt.figure(figsize=(6,4))
        sns.barplot(data=tdf, x='embedding', y='total_time', ci='sd')
        plt.title('Total run time by embedding')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'total_time_by_embedding.png'), dpi=200, bbox_inches='tight')
        plt.close()

    print('Saved plots to:', plot_dir)
except Exception as e:
    print('Plotting error (continuing):', e)

print('\nDone.')
