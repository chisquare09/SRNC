from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,adjusted_rand_score,f1_score,precision_score,recall_score
from mars.mars import MARS
from mars.experiment_dataset import ExperimentDataset
from args_parser import get_parser


mars_score = []

# Setting parameters
predictive_alg = "lightGBM"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2

# Experiment 1

# Experiment 2
control_neighbor = 5 
threshold_rejection = 0.7 
filter_proportion = 0 
data_set = ['bench']
# Experiment 3
control_neighbor = 5 
threshold_rejection = 0.3 
filter_proportion = 0 

# data_name = 'scPred'

adata_raw = sc.read_h5ad('/content/drive/MyDrive/mars/cellbench.h5ad')
adata_raw
adata = adata_raw.copy()

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.scale(adata, max_value=10, zero_center=True)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
sc.pp.pca(adata, n_comps=100)
sc.tl.tsne(adata)
sc.pl.tsne(adata, color=['experiment','ground_truth'],size=50)

adata_pca = AnnData(adata.obsm['X_pca'])
adata_pca.obs_names = adata.obs_names
adata_pca.obs['ground_truth'] = adata.obs['ground_truth']
adata_pca.obs['experiment'] = adata.obs['experiment']


fraction = 0.2 
def stratified_sample(X, y, fraction):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=fraction, random_state=42)
    for train_idx, sample_idx in sss.split(X, y):
        return X[sample_idx], y[sample_idx]
    
exp_10x = adata_pca[adata_pca.obs['experiment'] == '10x_5cl',:]
exp_celseq2 = adata_pca[adata_pca.obs['experiment'] == 'CelSeq2_5cl',:]

X_exp_10x = exp_10x.X.toarray()
y_exp_10x = np.array(exp_10x.obs['ground_truth'], dtype=np.int64)


X_celseq2 = exp_celseq2.X.toarray()
y_celseq2 = np.array(exp_celseq2.obs['ground_truth'], dtype=np.int64)

X_exp_10x_sampled, y_exp_10x_sampled = stratified_sample(X_exp_10x, y_exp_10x, fraction)
X_celseq2_sampled, y_celseq2_sampled = stratified_sample(X_celseq2, y_celseq2, fraction)

for data_name in data_set:
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

    # for seed in range(5):
    params, unknown = get_parser().parse_known_args()
    params
        # %%
    if torch.cuda.is_available() and not params.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device


    for i in np.unique(y_exp_10x_sampled):
        remove_label = i
        X_10x = X_exp_10x_sampled[y_exp_10x_sampled != remove_label]
        y_10x = y_exp_10x_sampled[y_exp_10x_sampled != remove_label]

        print("Data experiment...")
        print(f"Class removed from training:{i}")

        annotated_data = ExperimentDataset(X_10x, exp_10x.obs_names, exp_10x.var_names, '10x', y_10x)
        unannotated_data = ExperimentDataset(X_celseq2_sampled, exp_celseq2.obs_names, exp_celseq2.var_names, 'celseq2', y_celseq2_sampled)
        pretrain_data = ExperimentDataset(X_celseq2_sampled, exp_celseq2.obs_names, exp_celseq2.var_names, 'celseq2')
        n_clusters = len(np.unique(unannotated_data.y))

        # %% [markdown]
        # # Run MARS

        mars = MARS(n_clusters, params, [annotated_data], unannotated_data, pretrain_data, hid_dim_1=1000, hid_dim_2=100)

        # %%
        adata, landmarks, scores = mars.train(evaluation_mode=True, save_all_embeddings=False)
        print(adata)

        # %%
        scores
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame()
        df_score = pd.DataFrame([scores])
        df_score = df_score.drop(columns=['adj_mi', 'nmi'])
        mars_results_df = df_score



        # Specify the CSV file path
        csv_file = str(data_name) + '_mars_results'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'
        folder_path = '/content/drive/MyDrive/mars/results/mars_result'
        os.makedirs(folder_path, exist_ok=True)
        csv_file = os.path.join(folder_path, csv_file)

        if os.path.isfile(csv_file):
            # If the file exists, append the new data without the header
            df = pd.concat([df,df_score],ignore_index = True)
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            # If the file doesn't exist, write the new data with the header
            df_score.to_csv(csv_file, mode='w', header=True, index=False)



        # %%
        annotated_x = X_10x
        annotated_y = y_10x
        unannotated_x = X_celseq2_sampled
        unannotated_y = y_celseq2_sampled
        data_embbed_x=np.concatenate([annotated_x,unannotated_x])
        data_embbed_y=np.concatenate([annotated_y,unannotated_y])

        # %%
        y_all_labels = list(set(data_embbed_y))
    

        Y_predict_srnc=SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,
                                                control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)


        Y_predict_rejection=classification_rejection_v2(data_embbed_x,data_embbed_y,y_all_labels,annotated_x,annotated_y,unannotated_x,predictive_alg,threshold_rejection)


        ARI_overall_srnc_all.append(adjusted_rand_score(Y_predict_srnc,unannotated_y))
        accuracy_srnc_all.append(accuracy_score(Y_predict_srnc,unannotated_y))
        recall_unknown_srnc_all.append(recall_score(Y_predict_srnc,unannotated_y,average='weighted'))
        precision_unknown_srnc_all.append(precision_score(Y_predict_srnc,unannotated_y,average='weighted'))
        F1_unknown_srnc_all.append(f1_score(Y_predict_srnc,unannotated_y,average='weighted'))
        ARI_overall_rejection_all.append(adjusted_rand_score(Y_predict_rejection,unannotated_y))
        accuracy_rejection_all.append(accuracy_score(Y_predict_rejection,unannotated_y))
        recall_unknown_rejection_all.append(recall_score(Y_predict_rejection,unannotated_y,average='weighted'))
        precision_unknown_rejection_all.append(precision_score(Y_predict_rejection,unannotated_y,average='weighted'))
        F1_unknown_rejection_all.append(f1_score(Y_predict_rejection,unannotated_y,average='weighted'))

    # Create a DataFrame for storing results
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

    # Save the results to a CSV file
    folder_path2 ='/content/drive/MyDrive/mars/results/srnc_result'
    os.makedirs(folder_path2, exist_ok=True)
    result_file_name2 = str(data_name) + '_srnc_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'
    srnc_results_df.to_csv(os.path.join(folder_path2, result_file_name2), index=False)

    folder_path3 = '/content/drive/MyDrive/mars/results/rejection_result'
    os.makedirs(folder_path3, exist_ok=True)
    result_file_name3 = str(data_name) + '_rejection_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'
    rejection_results_df.to_csv(os.path.join(folder_path3,result_file_name3),index=False)

    # plot barchart and boxplot to compare 3 methods
    # barchart using sns library
    # average results of 3 methods
    mars_average_df = mars_results_df.mean(axis=0).to_dict()
    srnc_average_df = srnc_results_df.mean(axis=0).to_dict()
    rejection_average_df = rejection_results_df.mean(axis=0).to_dict()
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
    mars_results_df['method'] = 'MARS'
    srnc_results_df['method'] = 'srnc'
    rejection_results_df['method'] = 'rejection'
    result_df = pd.concat([mars_results_df,srnc_results_df,rejection_results_df],axis=0)
    box_plot_df = pd.melt(result_df,id_vars=['method'], value_vars=['adj_rand', 'accuracy', 'recall', 'precision', 'f1_score'],
                var_name='Metric', value_name='Value')
    sns.boxplot(x='Metric', y='Value', hue='method', data=box_plot_df, ax=axs[1])
    sns.despine(fig)
    axs[1].tick_params(bottom=True, left=True)
    sns.move_legend(axs[1],"upper left",bbox_to_anchor=(1,1))

    for ax in axs:
        ax.set_xlabel('')
    fig.tight_layout()
    # save plot
    folder_path4 = '/content/drive/MyDrive/mars/results/plot'
    os.makedirs(folder_path4, exist_ok=True)
    plt.savefig(os.path.join(folder_path4, f'Performance Metrics on {data_name} dataset {control_neighbor}_{threshold_rejection}_{filter_proportion}_{predictive_alg}.png'))
