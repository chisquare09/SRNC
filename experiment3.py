import time
from model.srnc import SequentialRadiusNeighborsClassifier
from model.rejection import classification_rejection_v2

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score,adjusted_rand_score,f1_score,precision_score,recall_score
from model.mars.mars import MARS
from model.mars.experiment_dataset import ExperimentDataset
from model.args_parser import get_parser
from model import rejection
from sklearn.decomposition import PCA

# Setting parameters
predictive_alg = "lightGBM"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 5 
threshold_rejection = 0.7 
filter_proportion = 0 
data_name = "PMBCs"

# Note: change the path to dataset
reference = pd.read_csv('./data/PMBCs/reference.csv', delimiter=',', header=None)
query = pd.read_csv('./data/PMBCs/query.csv', delimiter=',', header=None)

X_reference = reference.iloc[:,1:].to_numpy(dtype=float)
y_reference = reference.iloc[:, 0].to_numpy(dtype=int)
X_query = query.iloc[:,1:].to_numpy(dtype=float)
y_query = query.iloc[:, 0].to_numpy(dtype=int)

# Data preprocessing: PCA
pca = PCA(n_components=100)
X_reference = pca.fit_transform(X_reference)
X_query = pca.transform(X_query)

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

timings_list = []


params, unknown = get_parser().parse_known_args()
print(params)
if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Auto-detect best device (CUDA > CPU)
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f"[INFO] Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = 'mps'
    print(f"[INFO] Using MPS (Apple Silicon)")
else:
    device = 'cpu'
    print(f"[INFO] Using CPU")
print(f"Device: {device}")
params.device = device


# Loop to remove label randomly
for i in np.unique(y_reference):
    remove_label = i
    X_query = X_reference[y_reference != remove_label]
    y_query = y_reference[y_reference != remove_label].astype(dtype=int)

    num_cells_query, num_genes_query = X_query.shape
    num_cells_query, num_genes_query = X_query.shape

    print("Data experiment...")
    print(f"Class removed from training:{i}")

    annotated_data = ExperimentDataset(X_query, np.array([f"Cell {i}" for i in range(num_cells_query)]), np.array([f"Var {j}" for j in range(num_genes_query)]), 'query', y_query)
    unannotated_data = ExperimentDataset(X_query, np.array([f"Cell {i}" for i in range(num_cells_query)]), np.array([f"Var {j}" for j in range(num_genes_query)]), 'query', y_query)
    pretrain_data = ExperimentDataset(X_query, np.array([f"Cell {i}" for i in range(num_cells_query)]), np.array([f"Var {j}" for j in range(num_genes_query)]), 'query')
    n_clusters = len(np.unique(unannotated_data.y))

    # MARS implementation
    start_time_mars = time.time()
    mars = MARS(n_clusters, params, [annotated_data], unannotated_data, pretrain_data, hid_dim_1=1000, hid_dim_2=100)
    adata, landmarks, scores = mars.train(evaluation_mode=True, save_all_embeddings=False)
    end_time_mars = time.time()
    mars_time = end_time_mars - start_time_mars
    print(adata)
    print(scores)

    # Result dataframe
    df = pd.DataFrame()
    df_score = pd.DataFrame([scores])
    df_score = df_score.drop(columns=['adj_mi', 'nmi'])
    mars_results_df = df_score

    # mars_results_df = pd.DataFrame([scores]).drop(['adj_mi','nmi'])

    csv_file = str(data_name) + '_mars_results'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'PBMCs'+'.csv'
    # Note: change the path to save result
    folder_path = './results/experiment3/PBMCs/mars_result'
    os.makedirs(folder_path, exist_ok=True)
    csv_file = os.path.join(folder_path, csv_file)

    if os.path.isfile(csv_file):
        df = pd.concat([df,df_score],ignore_index = True)
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_score.to_csv(csv_file, mode='w', header=True, index=False)



    # SRNC and Rejection implementation
    annotated_x = X_query
    annotated_y = y_query
    unannotated_x = X_query
    unannotated_y = y_query
    data_embbed_x=np.concatenate([annotated_x,unannotated_x])
    data_embbed_y=np.concatenate([annotated_y,unannotated_y])

    y_all_labels = list(set(data_embbed_y))

    start_time_srnc = time.time()
    Y_predict_srnc=SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,
                                            control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)
    end_time_srnc = time.time()
    srnc_time = end_time_srnc - start_time_srnc

    start_time_rejection = time.time()
    Y_predict_rejection=rejection.classification_rejection_v2(data_embbed_x,data_embbed_y,y_all_labels,annotated_x,annotated_y,unannotated_x,predictive_alg,threshold_rejection)
    end_time_rejection = time.time()
    rejection_time = end_time_rejection - start_time_rejection

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

    timings_list.append({
        "data_name": data_name,
        "removed_label_sampled": remove_label,
        "mars_time": mars_time,
        "srnc_time": srnc_time,
        "rejection_time": rejection_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })


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



# Save the results to a CSV file

# Note: change the path to save result
folder_path2 ='./results/experiment3/PBMCs/srnc_result'
os.makedirs(folder_path2, exist_ok=True)
result_file_name2 = str(data_name) + '_srnc_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'PBMCs'+'.csv'
srnc_results_df.to_csv(os.path.join(folder_path2, result_file_name2), index=False)

# Note: change the path to save result
folder_path3 = './results/experiment3/PBMCs/rejection_result'
os.makedirs(folder_path3, exist_ok=True)
result_file_name3 = str(data_name) + '_rejection_result'+'_'+str(control_neighbor)+'_'+str(threshold_rejection)+'_'+str(filter_proportion)+predictive_alg+'PBMCs'+'.csv'
rejection_results_df.to_csv(os.path.join(folder_path3,result_file_name3),index=False)

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
# save plot

# Note: change the path to save result
folder_path4 = './results/experiment3/PBMCs/plot'
os.makedirs(folder_path4, exist_ok=True)
plt.savefig(os.path.join(folder_path4, f'Performance Metrics on {data_name} dataset {control_neighbor}_{threshold_rejection}_{filter_proportion}_{predictive_alg}_PBMCs.png'))


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
    out_dir = './results/experiment3/timings'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'timings_info_3.csv')
    timings_df.to_csv(csv_path, index=False)
    print(f'Wrote timings (including average row) to: {csv_path}')
else:
    print('timings_list is empty; no CSV written.')
