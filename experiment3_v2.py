#!/usr/bin/env python3
"""
experiment3_v2.py

Purpose:
 - Run the PMBCs experiment across multiple PCA component
 - For each removed label and for each n_pcs this script:
         * constructs ExperimentDataset objects using the first n_pcs PCA
             components,
         * trains/evaluates MARS (timed), runs SRNC (timed) and the Rejection
             classifier (timed),
         * collects performance metrics (ARI, Accuracy, Recall, Precision, F1)
             and timing information.

Outputs:
 - CSV summaries saved to `./results/experiment3/pca_compare/`:
         - `metrics_PMBCs.csv` (per-iteration metrics)
         - `timings_PMBCs.csv` (timings and an averaged row)
 - Plots are saved to `./results/experiment3/pca_compare/plots/`:
         - `{metric}_vs_pcs.png` for each metric, with separate lines per method
         - `time_by_method_vs_pcs.png` for per-method run time
"""

import os
import time
import gc
import random
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

# ---- Settings ----
pc_list = [5, 10, 20, 30,40, 50,60,70,80,90,100]
max_pc = max(pc_list)

# PMBCs CSV paths (as in your repo)
ref_path = './data/PMBCs/reference.csv'
query_path = './data/PMBCs/query.csv'

data_name = 'PMBCs'
out_dir = './results/experiment3/pca_compare'
plot_dir = os.path.join(out_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)


predictive_alg = "lightGBM"
shrink_parameter = 1
control_neighbor = 5
threshold_rejection = 0.7
filter_proportion = 0

# Read CSVs
reference = pd.read_csv(ref_path, delimiter=',', header=None)
query = pd.read_csv(query_path, delimiter=',', header=None)

X_reference = reference.iloc[:, 1:].to_numpy(dtype=float)
y_reference = reference.iloc[:, 0].to_numpy(dtype=int)
X_query = query.iloc[:, 1:].to_numpy(dtype=float)
y_query = query.iloc[:, 0].to_numpy(dtype=int)

# PCA fit on reference, transform both (preserves original experiment3 behavior)
print(f"Computing PCA on reference with n_components={max_pc} ...")
pca = PCA(n_components=max_pc)
X_reference_pca = pca.fit_transform(X_reference)
X_query_pca = pca.transform(X_query)

# Containers for results
metrics_rows = []
timings_rows = []

# CLI params & device
params, unknown = get_parser().parse_known_args()
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

# helper metrics
def safe_metrics(y_true, y_pred):
    return {
        'adj_rand': float(adjusted_rand_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }

# Follow experiment3.py: iterate over labels removed from reference
removed_labels_unique = np.unique(y_reference)
print("Removed labels to iterate:", removed_labels_unique)
iteration_idx = 0
for remove_label in removed_labels_unique:
    iteration_idx += 1
    print(f"\n=== Iteration {iteration_idx}: remove_label = {remove_label} ===")

    # Following original experiment3 logic: build X_query from reference with removed label
    X_query_loop = X_reference_pca[y_reference != remove_label]
    y_query_loop = y_reference[y_reference != remove_label].astype(dtype=int)

    num_cells_query, num_genes_query = X_query_loop.shape

    # For each n_pcs, slice PCA components and run the methods
    for n_pcs in pc_list:
        print(f"-- running n_pcs={n_pcs} --")
        X_q_n = X_query_loop[:, :n_pcs]

        # Create ExperimentDataset objects; keep same naming as experiment3
        annotated_cells = np.asarray([f"Cell {i}" for i in range(num_cells_query)])
        var_names = np.asarray([f"PC{j+1}" for j in range(n_pcs)])

        annotated_data = ExperimentDataset(np.asarray(X_q_n), annotated_cells, var_names, 'query', np.asarray(y_query_loop))
        unannotated_data = ExperimentDataset(np.asarray(X_q_n), annotated_cells, var_names, 'query', np.asarray(y_query_loop))
        pretrain_data = ExperimentDataset(np.asarray(X_q_n), annotated_cells, var_names, 'query')

        n_clusters = len(np.unique(unannotated_data.y))

        # MARS
        mars_time = None
        mars_metrics = {}
        try:
            t0 = time.perf_counter()
            mars = MARS(n_clusters, params, [annotated_data], unannotated_data, pretrain_data, hid_dim_1=1000, hid_dim_2=100)
            adata_out, landmarks, mars_scores = mars.train(evaluation_mode=True, save_all_embeddings=False)
            t1 = time.perf_counter()
            mars_time = t1 - t0
            if isinstance(mars_scores, dict):
                for k in ['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score']:
                    if k in mars_scores:
                        mars_metrics[k] = float(mars_scores[k])
        except Exception as e:
            print("MARS error:", e)
            mars_time = None
            mars_metrics = {}

        # SRNC & Rejection: use same features (PCA) as in this experiment
        # Build combined data for SRNC/Rejection
        # For experiment3's original behavior annotated_x == unannotated_x == X_q_n
        annotated_x = np.asarray(X_q_n)
        annotated_y = np.asarray(y_query_loop)
        unannotated_x = np.asarray(X_q_n)
        unannotated_y = np.asarray(y_query_loop)
        data_embbed_x = np.concatenate([annotated_x, unannotated_x])
        data_embbed_y = np.concatenate([annotated_y, unannotated_y])

        # SRNC (timed)
        srnc_time = None
        Y_predict_srnc = None
        try:
            t0 = time.perf_counter()
            Y_predict_srnc = SequentialRadiusNeighborsClassifier(data_embbed_x, list(set(data_embbed_y)), annotated_x, unannotated_x,
                                                                 annotated_y, predictive_alg, control_neighbor, shrink_parameter,
                                                                 filter_proportion, threshold_rejection)
            t1 = time.perf_counter()
            srnc_time = t1 - t0
        except Exception as e:
            print("SRNC error:", e)
            srnc_time = None

        # Rejection (timed)
        rejection_time = None
        Y_predict_rejection = None
        try:
            t0 = time.perf_counter()
            Y_predict_rejection = classification_rejection_v2(data_embbed_x, data_embbed_y, list(set(data_embbed_y)),
                                                              annotated_x, annotated_y, unannotated_x, predictive_alg, threshold_rejection)
            t1 = time.perf_counter()
            rejection_time = t1 - t0
        except Exception as e:
            print("Rejection error:", e)
            rejection_time = None

        # Metrics
        if Y_predict_srnc is not None:
            srnc_metrics = safe_metrics(unannotated_y, Y_predict_srnc)
        else:
            srnc_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        if Y_predict_rejection is not None:
            rej_metrics = safe_metrics(unannotated_y, Y_predict_rejection)
        else:
            rej_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

        # Append metrics rows
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

        # Append timings
        timings_rows.append({
            'data_name': data_name,
            'removed_label': int(remove_label),
            'n_pcs': n_pcs,
            'mars_time_s': mars_time,
            'srnc_time_s': srnc_time,
            'rejection_time_s': rejection_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })


# Finalize DataFrames
metrics_df = pd.DataFrame(metrics_rows)
timings_df = pd.DataFrame(timings_rows)

# Append averaged row to timings_df
if not timings_df.empty:
    timing_cols = ['mars_time_s', 'srnc_time_s', 'rejection_time_s']
    avg_row = {c: None for c in timings_df.columns}
    for c in timing_cols:
        if c in timings_df.columns:
            vals = pd.to_numeric(timings_df[c], errors='coerce')
            avg_row[c] = float(vals.mean()) if not vals.dropna().empty else None
    avg_row['data_name'] = data_name
    avg_row['removed_label'] = 'AVERAGE'
    avg_row['n_pcs'] = 'AVERAGE'
    timings_df = pd.concat([timings_df, pd.DataFrame([avg_row])], ignore_index=True)

# Save CSVs
metrics_csv = os.path.join(out_dir, f'metrics_{data_name}.csv')
timings_csv = os.path.join(out_dir, f'timings_{data_name}.csv')
metrics_df.to_csv(metrics_csv, index=False)
timings_df.to_csv(timings_csv, index=False)
print(f"Saved metrics -> {metrics_csv}")
print(f"Saved timings -> {timings_csv}")

# Plotting (aggregate lines per metric vs n_pcs)
try:
    plot_dir = plot_dir
    os.makedirs(plot_dir, exist_ok=True)

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
        fname = f"{metric}_vs_pcs.png"
        plt.savefig(os.path.join(plot_dir, fname), dpi=300)
        plt.close()

    # timings plot
    if 'mars_time_s' in timings_df.columns:
        timings_df['total_time_s'] = timings_df[['mars_time_s','srnc_time_s','rejection_time_s']].sum(axis=1, skipna=True)
        timings_plot_df = timings_df.copy()
        if 'n_pcs' in timings_plot_df.columns:
            timings_plot_df['n_pcs_numeric'] = pd.to_numeric(timings_plot_df['n_pcs'], errors='coerce')
            timings_plot_df = timings_plot_df.dropna(subset=['n_pcs_numeric'])
        else:
            timings_plot_df['n_pcs_numeric'] = None
        # per-method timing plot: melt timing columns and plot each method separately
        timing_cols = [c for c in ['mars_time_s', 'srnc_time_s', 'rejection_time_s'] if c in timings_plot_df.columns]
        if timing_cols:
            melt_df = timings_plot_df.melt(id_vars=['n_pcs_numeric'], value_vars=timing_cols, var_name='time_method', value_name='time_s')
            method_map = {
                'mars_time_s': 'MARS',
                'srnc_time_s': 'SRNC',
                'rejection_time_s': 'Rejection'
            }
            melt_df['method'] = melt_df['time_method'].map(method_map).fillna(melt_df['time_method'])
            melt_df = melt_df.dropna(subset=['time_s'])

            plt.figure(figsize=(6,4))
            sns.lineplot(data=melt_df, x='n_pcs_numeric', y='time_s', hue='method', estimator='mean', ci='sd', marker='o')
            plt.title('Method run time')
            plt.xlabel('PCs')
            plt.ylabel('Time (s)')
            plt.xticks(pc_list)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'time_by_method_vs_pcs.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print("Saved plots to:", plot_dir)
except Exception as e:
    print("Plotting error (continuing):", e)

print("Done.")
