"""
experiment1_v2.py

 - Changing threshold (0.0 .. 1.0 step 0.1) and comparing three methods:
     MARS, SRNC, and Rejection.
 - For each dataset and for each train/test split the script:
         * constructs ExperimentDataset objects for annotated and unannotated sets,
         * loops over threshold values and runs SRNC and Rejection with the
             current threshold, and also runs MARS (timed) for each threshold to
             collect per-threshold performance and runtime.
 - Collects per-threshold metrics (ARI, Accuracy, Recall, Precision, F1)
     and timings, writes CSV summaries under ./results/experiment1/thres_compare

"""

import time
from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,adjusted_rand_score,f1_score,precision_score,recall_score
import model.mars.mars as mars
from model.mars.mars import MARS
from model.mars.experiment_dataset import ExperimentDataset
# from model.mars.experiment_dataset import ExperimentDataset
from model.args_parser import get_parser


# Setting parameters
predictive_alg = "lightGBM"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 5 
threshold_rejection = 0.7
filter_proportion = 0 
data_set = ['pollen', 'patel', 'muraro', 'xin', 'zeisel', 'baron']
timings_list = []
# thresholds to sweep (0.0, 0.1, ..., 1.0)
thresholds = np.arange(0.0, 1.01, 0.1)
# loop through datasets
for data_name in data_set:
    if data_name=='muraro':
        k_fold = 3
    else:
        k_fold = 5

    if data_name=='pollen':
        # Note: change the path to benchmark dataset
        data = pd.read_csv('./data/benchmark/pollen-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='patel':
        data = pd.read_csv('./data/benchmark/patel-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='muraro':
        data = pd.read_csv('./data/benchmark/muraro-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='xin':
        data = pd.read_csv('./data/benchmark/xin-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='zeisel':
        data = pd.read_csv('./data/benchmark/zeisel-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='baron':
        data = pd.read_csv('./data/benchmark/baron-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    else:
        print("No data!")

 
    ARI_overall_srnc_all = []
    accuracy_srnc_all = []
    recall_unknown_srnc_all = []
    precision_unknown_srnc_all = []
    F1_unknown_srnc_all = []
    ARI_overall_rejection_all = []
    accuracy_rejection_all = []
    recall_unknown_rejection_all = []
    precision_unknown_rejection_all = []
    F1_unknown_rejection_all = []

    # Parameters for MARS
    params, unknown = get_parser().parse_known_args()
    print(params)
    if torch.cuda.is_available() and not params.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    print("Using device:", device)
    params.device = device

    # Data sampling: train and test set
    data = pd.DataFrame(data)
    annotated_data = pd.DataFrame()
    unannotated_data = pd.DataFrame()


    sss = StratifiedShuffleSplit(n_splits=k_fold,train_size=0.8,random_state=0)
    # containers for threshold-sweep results for this dataset
    metrics_threshold_rows = []
    timings_threshold_rows = []
    for train_index, test_index in sss.split(data.iloc[:,1:],data.iloc[:,0]):
        annotated_data = data.iloc[train_index]
        unannotated_data = data.iloc[test_index]

        # remove randomly 1 labels
        remove_label = np.random.choice(annotated_data.iloc[:,0].unique())
        annotated_data = annotated_data[annotated_data[0] != remove_label]
        print("Annotated Data:")
        print(annotated_data.shape)

        print("Unique values in annotated data: ", annotated_data.iloc[:,0].nunique())
        print("Class name: ", annotated_data.iloc[:,0].unique())
        print("Unannotated Data:")
        print(unannotated_data.shape)

        print("Unique values in unannotated data: ", unannotated_data.iloc[:,0].nunique())
        print("Class name: ", unannotated_data.iloc[:,0].unique())

        annotated = ad.AnnData(X=annotated_data.iloc[:,1:].values)
        annotated.obs['ground_truth'] = annotated_data.iloc[:,0].values

        annotated.obs['experiment'] = data_name
        unannotated = ad.AnnData(X=unannotated_data.iloc[:,1:].values)
        unannotated.obs['ground_truth'] = unannotated_data.iloc[:,0].values

        unannotated.obs['experiment'] = data_name
     
        annotated.obs_names = annotated_data.index.to_list()
        annotated.obs_names = [str(name) for name in annotated.obs_names]
        annotated.var_names = annotated_data.columns[1:]
        unannotated.obs_names = unannotated_data.index.to_list()
        unannotated.obs_names = [str(name) for name in unannotated.obs_names]
        unannotated.var_names = unannotated_data.columns[1:]

        # Train and Evaluate MARS

        annotated_y = np.array(annotated.obs['ground_truth'], dtype=np.int64)
        annotated_x = np.asarray(annotated.X)
        annotated_set = ExperimentDataset(annotated_x, annotated.obs_names, annotated.var_names, data_name, annotated_y)

        unannotated_y = np.array(unannotated.obs['ground_truth'], dtype=np.int64)
        unannotated_x = np.asarray(unannotated.X)
        unannotated_set = ExperimentDataset(unannotated_x, unannotated.obs_names, unannotated.var_names, data_name, unannotated_y)
        pretrain_data = ExperimentDataset(unannotated_x, unannotated.obs_names, unannotated.var_names, data_name)

        n_clusters = len(np.unique(unannotated_data[0]))
        print(n_clusters)

        # SRNC and Rejection implementation
        data_embbed_x=np.concatenate([annotated_x,unannotated_x])
        data_embbed_y=np.concatenate([annotated_y,unannotated_y])
        y_all_labels = list(set(data_embbed_y))

        # Run SRNC and rejection over the threshold sweep.
        # MARS will be executed inside the threshold loop so no pre-extraction here.

        for thr in thresholds:
            # MARS: run per-threshold as requested (threshold parameter not used by MARS but we run to collect timings per threshold)
            mars_metrics = {}
            mars_time = None
            try:
                t0_m = time.time()
                mars = MARS(n_clusters, params, [annotated_set], unannotated_set, pretrain_data, hid_dim_1=1000, hid_dim_2=100)
                out = mars.train(evaluation_mode=True, save_all_embeddings=False)
                t1_m = time.time()
                mars_time = t1_m - t0_m
                # handle returned tuple/dict variations
                if isinstance(out, tuple):
                    # try to unpack possible (adata, landmarks, scores) or (adata, scores)
                    if len(out) == 3:
                        _, _, scores = out
                    elif len(out) == 2:
                        _, scores = out
                    else:
                        scores = None
                else:
                    scores = out
                if isinstance(scores, dict):
                    for k in ['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score']:
                        if k in scores:
                            try:
                                mars_metrics[k] = float(scores[k])
                            except Exception:
                                mars_metrics[k] = None
                else:
                    mars_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            except Exception as e:
                print(f"MARS error at threshold={thr}:", e)
                mars_time = None
                mars_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

            # SRNC
            try:
                t0 = time.time()
                Y_predict_srnc = SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,
                                                        control_neighbor, shrink_parameter, filter_proportion, thr)
                srnc_time = time.time() - t0
            except Exception as e:
                print(f"SRNC error at threshold={thr}:", e)
                Y_predict_srnc = None
                srnc_time = None

            # Rejection
            try:
                t0 = time.time()
                Y_predict_rejection = classification_rejection_v2(data_embbed_x, data_embbed_y, y_all_labels, annotated_x, annotated_y, unannotated_x, predictive_alg, thr)
                rejection_time = time.time() - t0
            except Exception as e:
                print(f"Rejection error at threshold={thr}:", e)
                Y_predict_rejection = None
                rejection_time = None

            # compute metrics if predictions exist
            if Y_predict_srnc is not None:
                srnc_metrics = {
                    'adj_rand': float(adjusted_rand_score(Y_predict_srnc, unannotated_y)),
                    'accuracy': float(accuracy_score(Y_predict_srnc, unannotated_y)),
                    'recall': float(recall_score(Y_predict_srnc, unannotated_y, average='weighted')),
                    'precision': float(precision_score(Y_predict_srnc, unannotated_y, average='weighted')),
                    'f1_score': float(f1_score(Y_predict_srnc, unannotated_y, average='weighted'))
                }
            else:
                srnc_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

            if Y_predict_rejection is not None:
                rej_metrics = {
                    'adj_rand': float(adjusted_rand_score(Y_predict_rejection, unannotated_y)),
                    'accuracy': float(accuracy_score(Y_predict_rejection, unannotated_y)),
                    'recall': float(recall_score(Y_predict_rejection, unannotated_y, average='weighted')),
                    'precision': float(precision_score(Y_predict_rejection, unannotated_y, average='weighted')),
                    'f1_score': float(f1_score(Y_predict_rejection, unannotated_y, average='weighted'))
                }
            else:
                rej_metrics = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}

            # append metric rows for this threshold
            metrics_threshold_rows.append({
                'data_name': data_name,
                'removed_label_sampled': remove_label,
                'threshold': float(thr),
                'method': 'SRNC',
                **srnc_metrics
            })
            metrics_threshold_rows.append({
                'data_name': data_name,
                'removed_label_sampled': remove_label,
                'threshold': float(thr),
                'method': 'Rejection',
                **rej_metrics
            })
            # append MARS metrics for this threshold
            metrics_threshold_rows.append({
                'data_name': data_name,
                'removed_label_sampled': remove_label,
                'threshold': float(thr),
                'method': 'MARS',
                **{k: mars_metrics.get(k, None) for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            })

            timings_threshold_rows.append({
                'data_name': data_name,
                'removed_label_sampled': remove_label,
                'threshold': float(thr),
                'mars_time': mars_time,
                'srnc_time': srnc_time,
                'rejection_time': rejection_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })

    # Convert collected threshold metrics to DataFrame for summary/plots
    if metrics_threshold_rows:
        metrics_thresh_df = pd.DataFrame(metrics_threshold_rows)
    else:
        metrics_thresh_df = pd.DataFrame()

    # Save threshold CSVs and create per-threshold plots comparing methods
    try:
        thresh_out_dir = os.path.join('./results/experiment1/thres_compare', data_name)
        os.makedirs(thresh_out_dir, exist_ok=True)
        plot_dir_thresh = os.path.join(thresh_out_dir, 'plots')
        os.makedirs(plot_dir_thresh, exist_ok=True)

        if not metrics_thresh_df.empty:
            metrics_csv = os.path.join(thresh_out_dir, f'{data_name}_metrics_by_threshold.csv')
            metrics_thresh_df.to_csv(metrics_csv, index=False)

            # prepare for plotting
            plot_df = metrics_thresh_df.copy()
            plot_df['threshold_numeric'] = pd.to_numeric(plot_df['threshold'], errors='coerce')
            plot_df = plot_df.dropna(subset=['threshold_numeric'])
            metric_label_map = {'adj_rand': 'ARI', 'accuracy': 'Accuracy', 'recall': 'Recall', 'precision': 'Precision', 'f1_score': 'F1.score'}
            for metric in ['adj_rand','accuracy','recall','precision','f1_score']:
                display_name = metric_label_map.get(metric, metric)
                plt.figure(figsize=(6,4))
                sns.lineplot(data=plot_df, x='threshold_numeric', y=metric, hue='method', estimator='mean', ci='sd', marker='o')
                plt.title(f'{display_name} - {data_name}')
                plt.xlabel('Threshold')
                plt.ylabel(display_name)
                try:
                    plt.xticks(thresholds)
                except Exception:
                    pass
                try:
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if handles:
                        plt.legend(handles=handles, labels=labels, title='method', loc='center left', bbox_to_anchor=(1, 0.5))
                except Exception:
                    pass
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir_thresh, f'{metric}_vs_threshold.png'), dpi=200, bbox_inches='tight')
                plt.close()

        # timings
        if timings_threshold_rows:
            timings_thresh_df = pd.DataFrame(timings_threshold_rows)
            timings_csv = os.path.join(thresh_out_dir, f'{data_name}_timings_by_threshold.csv')
            timings_thresh_df.to_csv(timings_csv, index=False)

            tdf = timings_thresh_df.copy()
            tdf['threshold_numeric'] = pd.to_numeric(tdf['threshold'], errors='coerce')
            tdf = tdf.dropna(subset=['threshold_numeric'])
            time_cols = [c for c in ['mars_time','srnc_time','rejection_time'] if c in tdf.columns]
            if time_cols:
                melt = tdf.melt(id_vars=['threshold_numeric'], value_vars=time_cols, var_name='time_method', value_name='time_s')
                method_map = {'mars_time': 'MARS', 'srnc_time': 'SRNC', 'rejection_time': 'Rejection'}
                melt['method'] = melt['time_method'].map(method_map).fillna(melt['time_method'])
                melt = melt.dropna(subset=['time_s'])
                plt.figure(figsize=(6,4))
                sns.lineplot(data=melt, x='threshold_numeric', y='time_s', hue='method', estimator='mean', ci='sd', marker='o')
                plt.title(f'{data_name}')
                plt.xlabel('Threshold')
                plt.ylabel('Time (s)')
                try:
                    plt.xticks(thresholds)
                except Exception:
                    pass
                try:
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if handles:
                        plt.legend(handles=handles, labels=labels, title='method', loc='center left', bbox_to_anchor=(1, 0.5))
                except Exception:
                    pass
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir_thresh, 'time_by_method_vs_threshold.png'), dpi=200, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print('Threshold plotting/saving error:', e)

    # Result dataframes
    srnc_results_df = pd.DataFrame({
        "adj_rand": ARI_overall_srnc_all,
        "accuracy": accuracy_srnc_all,
        "recall": recall_unknown_srnc_all,
        "precision": precision_unknown_srnc_all,
        "f1_score": F1_unknown_srnc_all,
    })

    rejection_results_df = pd.DataFrame({
        "adj_rand": ARI_overall_rejection_all,
        "accuracy": accuracy_rejection_all,
        "recall": recall_unknown_rejection_all,
        "precision": precision_unknown_rejection_all,
        "f1_score": F1_unknown_rejection_all,
    })

    result_file_name2 = str(data_name) + '_srnc_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'
    
    # Note: change the path to save result
    root_path2='./results/experiment1/thres_compare/srnc_result'
    os.makedirs(os.path.join(root_path2, data_name), exist_ok=True)


    srnc_results_df.to_csv(os.path.join(root_path2,data_name, result_file_name2), index=False)

    result_file_name3 = str(data_name) + '_rejection_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'

    # Note: change the path to save result
    root_path3='./results/experiment1/thres_compare/rejection_result'
    os.makedirs(os.path.join(root_path3, data_name), exist_ok=True)
    rejection_results_df.to_csv(os.path.join(root_path3,data_name,result_file_name3),index=False)

    # Plot the results: derive averages from threshold-aggregated metrics if present
    if not metrics_thresh_df.empty:
        try:
            mars_df = metrics_thresh_df[metrics_thresh_df['method'] == 'MARS']
            srnc_df = metrics_thresh_df[metrics_thresh_df['method'] == 'SRNC']
            rej_df = metrics_thresh_df[metrics_thresh_df['method'] == 'Rejection']
            mars_average_df = mars_df[['adj_rand','accuracy','recall','precision','f1_score']].mean(axis=0).to_dict() if not mars_df.empty else {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            srnc_average_df = srnc_df[['adj_rand','accuracy','recall','precision','f1_score']].mean(axis=0).to_dict() if not srnc_df.empty else {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            rejection_average_df = rej_df[['adj_rand','accuracy','recall','precision','f1_score']].mean(axis=0).to_dict() if not rej_df.empty else {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
        except Exception:
            mars_average_df = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            srnc_average_df = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
            rejection_average_df = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
    else:
        mars_average_df = {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
        srnc_average_df = srnc_results_df.mean(axis=0).to_dict() if not srnc_results_df.empty else {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
        rejection_average_df = rejection_results_df.mean(axis=0).to_dict() if not rejection_results_df.empty else {k: None for k in ['adj_rand','accuracy','recall','precision','f1_score']}
    mars_average_df['method'] = 'MARS'
    srnc_average_df['method'] = 'srnc'
    rejection_average_df['method'] = 'rejection'
    average_df = pd.DataFrame([mars_average_df,srnc_average_df,rejection_average_df])
    print(average_df)

    fig, axs = plt.subplots(2,1)
    sns.set_theme(style="white")

    # figure 1
    bar_plot_df = pd.melt(average_df,id_vars=['method'],value_vars=['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score'],
                        var_name='Metric',value_name='Value')
    sns.barplot(x='Metric',y='Value',hue='method',data=bar_plot_df,errorbar=None,ax=axs[0])
    sns.despine(fig)
    axs[0].tick_params(bottom=True, left=True)
    sns.move_legend(axs[0],"upper left",bbox_to_anchor=(1,1))
    axs[0].set_title(f'Performance Metrics on {data_name} dataset {control_neighbor}_{threshold_rejection}_{filter_proportion}_{predictive_alg}')
    axs[0].set_ylim(0, 1)

    # figure 2
    # For boxplots use the threshold-aggregated per-method data when available
    if not metrics_thresh_df.empty:
        mars_results_df_plot = metrics_thresh_df[metrics_thresh_df['method'] == 'MARS'].copy()
        srnc_results_df_plot = metrics_thresh_df[metrics_thresh_df['method'] == 'SRNC'].copy()
        rejection_results_df_plot = metrics_thresh_df[metrics_thresh_df['method'] == 'Rejection'].copy()
    else:
        mars_results_df_plot = pd.DataFrame(columns=['adj_rand','accuracy','recall','precision','f1_score'])
        srnc_results_df_plot = srnc_results_df.copy()
        rejection_results_df_plot = rejection_results_df.copy()

    mars_results_df_plot['method'] = 'MARS'
    srnc_results_df_plot['method'] = 'srnc'
    rejection_results_df_plot['method'] = 'rejection'
    result_df = pd.concat([mars_results_df_plot, srnc_results_df_plot, rejection_results_df_plot], axis=0)
    box_plot_df = pd.melt(result_df,id_vars=['method'], value_vars=['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score'],
                var_name='Metric', value_name='Value')
    sns.boxplot(x='Metric', y='Value', hue='method', data=box_plot_df, ax=axs[1])
    sns.despine(fig)
    axs[1].tick_params(bottom=True, left=True)
    sns.move_legend(axs[1],"upper left",bbox_to_anchor=(1,1))

    for ax in axs:
        ax.set_xlabel('')
    fig.tight_layout()
    fig_name = f'Performance Metrics on {data_name} dataset {control_neighbor}_{threshold_rejection}_{filter_proportion}_{predictive_alg}.png'
    # Note: change the path to save result
    root_path4='./results/experiment1/thres_compare/plot'
    os.makedirs(os.path.join(root_path4, data_name), exist_ok=True)
    plt.savefig(os.path.join(root_path4, data_name,fig_name))


if len(timings_list) > 0:
    timings_df = pd.DataFrame(timings_list)
    # names of timing columns to average
    timing_cols = ['mars_time', 'srnc_time', 'rejection_time']

    # make sure the timing columns exist (skip missing ones)
    timing_cols = [c for c in timing_cols if c in timings_df.columns]

    # compute averages (NaN-safe)
    avg_values = {c: float(timings_df[c].mean()) if c in timing_cols else np.nan for c in timings_df.columns}

    # set non-timing columns explicitly to None (null in CSV)
    for c in timings_df.columns:
        if c not in timing_cols:
            avg_values[c] = None

    # optional: mark a descriptive data_name / row-type
    if 'data_name' in timings_df.columns:
        avg_values['data_name'] = 'AVERAGE'

    # append final averaged row
    timings_df = pd.concat([timings_df, pd.DataFrame([avg_values])], ignore_index=True)

    # write CSV (replace path as needed)
    out_dir = './results/experiment1/thres_compare/timings'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'timings_info_1.csv')
    timings_df.to_csv(csv_path, index=False)
    print(f'Wrote timings (including average row) to: {csv_path}')
else:
    print('timings_list is empty; no CSV written.')
