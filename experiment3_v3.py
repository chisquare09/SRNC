#!/usr/bin/env python3
"""
experiment3_v3.py

Compare PCA(50) vs scVI(50) embeddings on the PMBCs reference/query setup.

For each label removed from the reference set the script:
 - builds embeddings (PCA on reference; optional scVI trained on combined ref+query),
 - for each embedding (PCA50, scVI) runs MARS, SRNC and Rejection (timed),
 - collects metrics and timings and writes CSVs/plots to
   ./results/experiment3/compare_scvi_pca/

Notes:
 - scvi-tools is optional; if unavailable the script runs PCA50 only.
 - scVI training can be slow; use small epochs for quick smoke tests.
"""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, precision_score, recall_score

from model.mars.mars import MARS
from model.mars.experiment_dataset import ExperimentDataset
from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2
from model.args_parser import get_parser

import anndata as ad

# Settings
pc_n = 50
latent_dim_scvi = 50
ref_path = './data/PMBCs/reference.csv'
query_path = './data/PMBCs/query.csv'
out_dir = './results/experiment3/compare_scvi_pca'
os.makedirs(out_dir, exist_ok=True)

predictive_alg = 'lightGBM'
control_neighbor = 5
threshold_rejection = 0.7
filter_proportion = 0
shrink_parameter = 1

# parse args
params, unknown = get_parser().parse_known_args()

# device
if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print('Using device:', device)
params.device = device

# helper
def safe_metrics(y_true, y_pred):
    return {
        'adj_rand': float(adjusted_rand_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }

# Read CSVs
print('Loading reference/query CSVs')
ref_df = pd.read_csv(ref_path, header=None)
query_df = pd.read_csv(query_path, header=None)

X_ref = ref_df.iloc[:, 1:].to_numpy(dtype=float)
y_ref = ref_df.iloc[:, 0].to_numpy()
X_query = query_df.iloc[:, 1:].to_numpy(dtype=float)
y_query = query_df.iloc[:, 0].to_numpy()

# PCA fit on reference and transform query
print(f'Computing PCA on reference (n_components={pc_n})')
pca = PCA(n_components=pc_n)
X_ref_pca = pca.fit_transform(X_ref)
X_query_pca = pca.transform(X_query)

# Optional scVI
scvi_available = True
try:
    import scvi
    from scvi.model import SCVI
except Exception:
    scvi_available = False
    print('scvi-tools not available; scVI embedding will be skipped.')

if scvi_available:
    try:
        print('Training scVI on combined ref+query (may take time)')
        adata_all = ad.AnnData(np.vstack([X_ref, X_query]))
        scvi.model.SCVI.setup_anndata(adata_all)
        n_epochs = getattr(params, 'epochs', 50) or 50
        scvi_model = SCVI(adata_all, n_latent=latent_dim_scvi)
        scvi_model.train(max_epochs=int(n_epochs))
        latent_all = scvi_model.get_latent_representation()
        X_ref_scvi = latent_all[: X_ref.shape[0]]
        X_query_scvi = latent_all[X_ref.shape[0] :]
    except Exception as e:
        print('scVI training failed, skipping scVI embedding:', e)
        scvi_available = False

# Containers
metrics_rows = []
timings_rows = []

removed_labels_unique = np.unique(y_ref)
print('Removed labels to iterate:', removed_labels_unique)
iter_idx = 0
for remove_label in removed_labels_unique:
    iter_idx += 1
    print(f'\n=== Iter {iter_idx}: remove_label = {remove_label} ===')

    # following experiment3 logic: build X_query_loop from reference without remove_label
    X_q_pca = X_ref_pca[y_ref != remove_label]
    y_q = y_ref[y_ref != remove_label].astype(int)

    if scvi_available:
        X_q_scvi = X_ref_scvi[y_ref != remove_label]

    num_cells = X_q_pca.shape[0]
    annotated_cells = np.asarray([f'Cell {i}' for i in range(num_cells)])

    for embedding_name in (['PCA50'] + (['scVI'] if scvi_available else [])):
        print(f'-- embedding: {embedding_name} --')
        if embedding_name == 'PCA50':
            X_n = X_q_pca[:, :pc_n]
            var_names = np.asarray([f'PC{j+1}' for j in range(pc_n)])
        else:
            X_n = X_q_scvi[:, :latent_dim_scvi]
            var_names = np.asarray([f'LAT{j+1}' for j in range(latent_dim_scvi)])

        annotated_data = ExperimentDataset(np.asarray(X_n), annotated_cells, var_names, 'query', np.asarray(y_q))
        unannotated_data = ExperimentDataset(np.asarray(X_n), annotated_cells, var_names, 'query', np.asarray(y_q))
        pretrain_data = ExperimentDataset(np.asarray(X_n), annotated_cells, var_names, 'query')

        n_clusters = len(np.unique(unannotated_data.y))

        # MARS
        mars_time = None
        mars_metrics = {}
        try:
            t0 = time.perf_counter()
            mars = MARS(n_clusters, params, [annotated_data], unannotated_data, pretrain_data, hid_dim_1=1000, hid_dim_2=100)
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

        # SRNC & Rejection (use same X_n as features)
        data_embbed_x = np.concatenate([np.asarray(X_n), np.asarray(X_n)])
        data_embbed_y = np.concatenate([np.asarray(y_q), np.asarray(y_q)])
        annotated_x = np.asarray(X_n)
        unannotated_x = np.asarray(X_n)
        annotated_y = np.asarray(y_q)

        # SRNC
        try:
            t0 = time.perf_counter()
            Y_pred_srnc = SequentialRadiusNeighborsClassifier(data_embbed_x, list(set(data_embbed_y)), annotated_x, unannotated_x, annotated_y, predictive_alg, control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)
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

        if Y_pred_srnc is not None:
            srnc_metrics = safe_metrics(annotated_y, Y_pred_srnc)
        else:
            srnc_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        if Y_pred_rej is not None:
            rej_metrics = safe_metrics(annotated_y, Y_pred_rej)
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

# Save CSVs
metrics_df = pd.DataFrame(metrics_rows)
timings_df = pd.DataFrame(timings_rows)
metrics_csv = os.path.join(out_dir, 'metrics_scvi_vs_pca50_PMBCs.csv')
timings_csv = os.path.join(out_dir, 'timings_scvi_vs_pca50_PMBCs.csv')
metrics_df.to_csv(metrics_csv, index=False)
timings_df.to_csv(timings_csv, index=False)
print('Saved metrics ->', metrics_csv)
print('Saved timings ->', timings_csv)

# Plotting
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
        plt.savefig(os.path.join(plot_dir, f'{metric}_by_removed_label_PMBCs.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # timings
    if not timings_df.empty:
        tdf = timings_df.copy()
        tdf['total_time'] = tdf[['mars_time_s','srnc_time_s','rejection_time_s']].sum(axis=1, skipna=True)
        plt.figure(figsize=(6,4))
        sns.barplot(data=tdf, x='embedding', y='total_time', ci='sd')
        plt.title('Total run time')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'total_time_by_embedding_PMBCs.png'), dpi=200, bbox_inches='tight')
        plt.close()

    print('Saved plots to:', plot_dir)
except Exception as e:
    print('Plotting error (continuing):', e)

print('\nAll done.')
