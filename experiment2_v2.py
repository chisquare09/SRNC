#!/usr/bin/env python3
"""
experiment2_v2.py

PCA change over multiple numbers of components (5,10,20,30,50,100).
For each removed class (as in your original script) and each n_pcs:
 - build ExperimentDataset from the first n_pcs PCA components
 - run MARS (timed)
 - run SRNC (timed)
 - run rejection (timed)
 - compute metrics and collect timings
Results are written to ./results/experiment2/pca_compare
"""

import os
import time
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import torch

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from model.mars.mars import MARS
from model.mars.experiment_dataset import ExperimentDataset
from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2
from model.args_parser import get_parser

# ---- Settings ----
pc_list = [5, 10, 20, 30,40, 50,60,70,80,90,100]
max_pc = max(pc_list)
data_h5ad = './data/bench/cellbench.h5ad'
data_name = 'bench'
out_dir = './results/experiment2/pca_compare'
os.makedirs(out_dir, exist_ok=True)

# Experiment hyperparams (can adjust or pass via args later)
control_neighbor = 5
threshold_rejection = 0.7
filter_proportion = 5
predictive_alg = "lightGBM"
shrink_parameter = 1

# Parse CLI args (reuses your parser for epochs, pretrain, batch etc)
params, unknown = get_parser().parse_known_args()

# Device selection
if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available():
    device = 'cuda:0'
    print("[INFO] Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("[INFO] Using MPS (Apple Silicon)")
else:
    device = 'cpu'
    print("[INFO] Using CPU")
params.device = device

# Debug / reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# ---- Load and preprocess ----
print("Loading data:", data_h5ad)
adata_raw = sc.read_h5ad(data_h5ad)
print(adata_raw)

# keep original counts layer if present
try:
    adata_raw.layers['counts'] = adata_raw.X.copy()
except Exception:
    pass

batch_key = 'experiment'
label_key = 'ground_truth'

adata = adata_raw.copy()

# HVG (as in your original script)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key=batch_key)
adata_hvg = adata[:, adata.var['highly_variable']].copy()

# Compute PCA once with max_pc
print(f"Computing PCA with n_comps={max_pc} ...")
sc.pp.pca(adata_hvg, n_comps=max_pc)
X_pca_full = adata_hvg.obsm['X_pca']  # numpy array shape (n_cells, max_pc)
labels_all = adata_hvg.obs[label_key].values
experiments_all = adata_hvg.obs[batch_key].values
cell_names_all = np.array(adata_hvg.obs_names)

# Identify masks for the experiments you use in the original script
mask_10x = experiments_all == '10x_5cl'
mask_celseq2 = experiments_all == 'CelSeq2_5cl'

X_exp_10x_pca = X_pca_full[mask_10x]
y_exp_10x = labels_all[mask_10x]
cells_exp_10x = cell_names_all[mask_10x]

X_celseq2_pca = X_pca_full[mask_celseq2]
y_celseq2 = labels_all[mask_celseq2]
cells_celseq2 = cell_names_all[mask_celseq2]

print("Shapes: 10x PCA:", X_exp_10x_pca.shape, "celseq2 PCA:", X_celseq2_pca.shape)

# Containers for collected results
metrics_rows = []   # will contain dicts with data_name, n_pcs, removed_label, method, fold(=iteration) and metrics
timings_rows = []   # will contain dicts with data_name, n_pcs, removed_label, mars_time_s, srnc_time_s, rejection_time_s

# Helper to compute metrics
def safe_metrics(y_true, y_pred):
    return {
        'adj_rand': float(adjusted_rand_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }

# Main loop: for each removed label (as in original) and for each PCA dimension
removed_labels_unique = np.unique(y_exp_10x)
print("Removed labels to iterate:", removed_labels_unique)

iteration_idx = 0
for remove_label in removed_labels_unique:
    iteration_idx += 1
    print(f"\n=== Iteration {iteration_idx}: remove_label = {remove_label} ===")

    # Build training set (10x) with the chosen label removed
    train_mask = y_exp_10x != remove_label
    if np.sum(train_mask) == 0:
        print("Warning: removing this label leaves zero training examples; skipping.")
        continue

    for n_pcs in pc_list:
        print(f"-- running n_pcs={n_pcs} --")
        # slice PCA components
        X_10x = X_exp_10x_pca[:, :n_pcs]   # full 10x subset for this n_pcs
        X_10x_train = X_10x[train_mask]
        y_10x_train = y_exp_10x[train_mask]

        # unannotated is always the celseq2 subset
        X_celseq2 = X_celseq2_pca[:, :n_pcs]
        y_celseq2_full = y_celseq2

        # Create ExperimentDataset objects (use original names if desired)
        # pass numpy arrays for cells/genes so ExperimentDataset can call .tolist() if needed
        annotated_cells = np.asarray(cells_exp_10x[train_mask])
        unannotated_cells = np.asarray(cells_celseq2)
        var_names = np.asarray([f'PC{i+1}' for i in range(n_pcs)])

        annotated_set = ExperimentDataset(np.asarray(X_10x_train), annotated_cells, var_names, '10x', np.asarray(y_10x_train))
        unannotated_set = ExperimentDataset(np.asarray(X_celseq2), unannotated_cells, var_names, 'celseq2', np.asarray(y_celseq2_full))
        pretrain_set = ExperimentDataset(np.asarray(X_celseq2), unannotated_cells, var_names, 'celseq2')

        # Run MARS (timed)
        mars_time = None
        mars_metrics = {}
        try:
            t0 = time.perf_counter()
            mars = MARS(len(np.unique(unannotated_set.y)), params, [annotated_set], unannotated_set, pretrain_set,
                        hid_dim_1=1000, hid_dim_2=100)
            adata_out, landmarks, mars_scores = mars.train(evaluation_mode=True, save_all_embeddings=False)
            t1 = time.perf_counter()
            mars_time = t1 - t0
            # if mars_scores is a dict, try extract metrics (may vary)
            if isinstance(mars_scores, dict):
                for k in ['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score']:
                    if k in mars_scores:
                        mars_metrics[k] = float(mars_scores[k])
        except Exception as e:
            print("MARS error:", e)
            mars_time = None
            mars_metrics = {}

        # Build combined data for SRNC/Rejection (they expect original features but PCA is used here)
        data_embbed_x = np.concatenate([np.asarray(X_10x_train), np.asarray(X_celseq2)])
        data_embbed_y = np.concatenate([np.asarray(y_10x_train), np.asarray(y_celseq2_full)])

        # SRNC (timed)
        srnc_time = None
        Y_pred_srnc = None
        try:
            t0 = time.perf_counter()
            Y_pred_srnc = SequentialRadiusNeighborsClassifier(data_embbed_x, list(set(data_embbed_y)),
                                                             np.asarray(X_10x_train), np.asarray(X_celseq2),
                                                             np.asarray(y_10x_train), predictive_alg,
                                                             control_neighbor, shrink_parameter, filter_proportion,
                                                             threshold_rejection)
            t1 = time.perf_counter()
            srnc_time = t1 - t0
        except Exception as e:
            print("SRNC error:", e)
            srnc_time = None

        # Rejection (timed)
        rej_time = None
        Y_pred_rej = None
        try:
            t0 = time.perf_counter()
            Y_pred_rej = classification_rejection_v2(data_embbed_x, data_embbed_y, list(set(data_embbed_y)),
                                                     np.asarray(X_10x_train), np.asarray(y_10x_train),
                                                     np.asarray(X_celseq2), predictive_alg, threshold_rejection)
            t1 = time.perf_counter()
            rej_time = t1 - t0
        except Exception as e:
            print("Rejection error:", e)
            rej_time = None

        # Compute metrics (if predictions available)
        if Y_pred_srnc is not None:
            srnc_metrics = safe_metrics(y_celseq2_full, Y_pred_srnc)
        else:
            srnc_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        if Y_pred_rej is not None:
            rej_metrics = safe_metrics(y_celseq2_full, Y_pred_rej)
        else:
            rej_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        # Store metrics rows for SRNC, Rejection and (optionally) MARS
        metrics_rows.append({
            'data_name': data_name,
            'removed_label': int(remove_label),
            'n_pcs': n_pcs,
            'method': 'SRNC',
            **srnc_metrics
        })
        metrics_rows.append({
            'data_name': data_name,
            'removed_label': int(remove_label),
            'n_pcs': n_pcs,
            'method': 'Rejection',
            **rej_metrics
        })
        if mars_metrics:
            metrics_rows.append({
                'data_name': data_name,
                'removed_label': int(remove_label),
                'n_pcs': n_pcs,
                'method': 'MARS',
                **{k: mars_metrics.get(k, None) for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            })

        # Store timing row
        timings_rows.append({
            'data_name': data_name,
            'removed_label': int(remove_label),
            'n_pcs': n_pcs,
            'mars_time_s': mars_time,
            'srnc_time_s': srnc_time,
            'rejection_time_s': rej_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })


# Convert to DataFrames and save
metrics_df = pd.DataFrame(metrics_rows)
timings_df = pd.DataFrame(timings_rows)

# Append average row to timings_df (averages for timing columns; other columns set to None)
if not timings_df.empty:
    timing_cols = ['mars_time_s', 'srnc_time_s', 'rejection_time_s']
    avg_row = {c: None for c in timings_df.columns}
    for c in timing_cols:
        if c in timings_df.columns:
            # ignore None values (coerce to numeric)
            vals = pd.to_numeric(timings_df[c], errors='coerce')
            avg_val = float(vals.mean()) if not vals.dropna().empty else None
            avg_row[c] = avg_val
    avg_row['data_name'] = data_name
    avg_row['removed_label'] = 'AVERAGE'
    avg_row['n_pcs'] = 'AVERAGE'
    timings_df = pd.concat([timings_df, pd.DataFrame([avg_row])], ignore_index=True)

# Save CSVs locally
metrics_csv = os.path.join(out_dir, f'metrics_{data_name}.csv')
timings_csv = os.path.join(out_dir, f'timings_{data_name}.csv')
metrics_df.to_csv(metrics_csv, index=False)
timings_df.to_csv(timings_csv, index=False)
print(f"Saved metrics -> {metrics_csv}")
print(f"Saved timings -> {timings_csv}")

# Simple plotting of aggregated results
try:
    agg = metrics_df.groupby(['method','n_pcs']).agg(
        adj_rand_mean=('adj_rand','mean'), adj_rand_std=('adj_rand','std'),
        accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'),
        recall_mean=('recall','mean'), recall_std=('recall','std'),
        precision_mean=('precision','mean'), precision_std=('precision','std'),
        f1_mean=('f1_score','mean'), f1_std=('f1_score','std'),
    ).reset_index()

    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # line plots for each metric (use friendly display names)
    metric_label_map = {
        'adj_rand': 'ARI',
        'accuracy': 'Accuracy',
        'recall': 'Recall',
        'precision': 'Precision',
        'f1_score': 'F1.score'
    }

    for metric in ['adj_rand','accuracy','recall','precision','f1_score']:
        display_name = metric_label_map.get(metric, metric)
        plt.figure(figsize=(6,4))
        sns.lineplot(data=metrics_df, x='n_pcs', y=metric, hue='method', estimator='mean', ci='sd', marker='o')
        plt.title(f'{display_name}')
        plt.xlabel('PCs')
        plt.xticks(pc_list)
        plt.ylabel(display_name)
        plt.legend(title='method', loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        fname = f"{metric}_vs_pcs.png"  # keep original metric key in filename
        plt.savefig(os.path.join(plot_dir, fname), dpi=300)
        plt.close()

    # timings: total time vs n_pcs
    if 'mars_time_s' in timings_df.columns:
        timings_df['total_time_s'] = timings_df[['mars_time_s','srnc_time_s','rejection_time_s']].sum(axis=1, skipna=True)
        timings_plot_df = timings_df.copy()
        if 'n_pcs' in timings_plot_df.columns:
            timings_plot_df['n_pcs_numeric'] = pd.to_numeric(timings_plot_df['n_pcs'], errors='coerce')
            timings_plot_df = timings_plot_df.dropna(subset=['n_pcs_numeric'])
        else:
            timings_plot_df['n_pcs_numeric'] = None

        plt.figure(figsize=(6,4))
        sns.lineplot(data=timings_plot_df, x='n_pcs_numeric', y='total_time_s', estimator='mean', ci='sd', marker='o')
        plt.title('Run time vs PCA components')
        plt.xlabel('PCs')
        plt.xticks(pc_list)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'time_total_vs_pcs.png'), dpi=300)
        plt.close()

    print("Saved plots to:", plot_dir)
except Exception as e:
    print("Plotting error (continuing):", e)

print("Done.")