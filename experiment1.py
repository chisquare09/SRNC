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
from mars.mars import MARS
from mars.experiment_dataset import ExperimentDataset
from args_parser import get_parser

# Setting parameters
predictive_alg = "lightGBM"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 5 
threshold_rejection = 0.3 
filter_proportion = 0 
data_set = ['pollen', 'patel', 'muraro', 'xin', 'zeisel', 'baron']

# loop through datasets
for data_name in data_set:
    if data_name=='muraro':
        k_fold = 3
    else:
        k_fold = 5

    if data_name=='pollen':
        # Note: change the path to benchmark dataset
        data = pd.read_csv('path/to/pollen-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='patel':
        data = pd.read_csv('path/to/patel-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='muraro':
        data = pd.read_csv('path/to/muraro-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='xin':
        data = pd.read_csv('path/to/xin-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='zeisel':
        data = pd.read_csv('path/to/zeisel-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
    elif data_name=='baron':
        data = pd.read_csv('path/to/baron-prepare-log_count_100pca.csv',delimiter=',',header=None, dtype='float32')
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
    params
    if torch.cuda.is_available() and not params.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device

    # Data sampling: train and test set
    data = pd.DataFrame(data)
    annotated_data = pd.DataFrame()
    unannotated_data = pd.DataFrame()


    sss = StratifiedShuffleSplit(n_splits=k_fold,train_size=0.8,random_state=0)
    for train_index, test_index in sss.split(data.iloc[:,1:],data.iloc[:,0]):
        annotated_data = data.iloc[train_index]
        unannotated_data = data.iloc[test_index]

        # remove randomly 1 labels
        annotated_data = annotated_data[annotated_data[0] != np.random.choice(annotated_data.iloc[:,0].unique())]
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
        mars = MARS(n_clusters, params, [annotated_set], unannotated_set, pretrain_data, hid_dim_1=1000, hid_dim_2=100)
        adata, landmarks, scores = mars.train(evaluation_mode=True, save_all_embeddings=False)
        print(adata)

        scores
        # Result dataframe
        df = pd.DataFrame()
        df_score = pd.DataFrame([scores])
        df_score = df_score.drop(columns=['adj_mi', 'nmi'])
        mars_results_df = df_score



        # Save the results to a CSV file
        csv_file = str(data_name) + '_mars_results'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'
        
        # Note: change the path to save result
        root_path1 = 'path/to/results/mars_result'
        os.makedirs(os.path.join(root_path1, data_name), exist_ok=True)
        csv_file = os.path.join(root_path1,data_name, csv_file)

        if os.path.isfile(csv_file):
            df = pd.concat([df,df_score],ignore_index = True)
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_score.to_csv(csv_file, mode='w', header=True, index=False)

        # SRNC and Rejection implementation
        data_embbed_x=np.concatenate([annotated_x,unannotated_x])
        data_embbed_y=np.concatenate([annotated_y,unannotated_y])
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
    root_path2='path/to/results/srnc_result'
    os.makedirs(os.path.join(root_path2, data_name), exist_ok=True)


    srnc_results_df.to_csv(os.path.join(root_path2,data_name, result_file_name2), index=False)

    result_file_name3 = str(data_name) + '_rejection_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'.csv'

    # Note: change the path to save result
    root_path3='path/to/results/rejection_result'
    os.makedirs(os.path.join(root_path3, data_name), exist_ok=True)
    rejection_results_df.to_csv(os.path.join(root_path3,data_name,result_file_name3),index=False)

    # Plot the results
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
    fig_name = f'Performance Metrics on {data_name} dataset {control_neighbor}_{threshold_rejection}_{filter_proportion}_{predictive_alg}.png'
    # Note: change the path to save result
    root_path4='path/to/results/plot'
    os.makedirs(os.path.join(root_path4, data_name), exist_ok=True)
    plt.savefig(os.path.join(root_path4, data_name,fig_name))
