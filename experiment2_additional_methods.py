#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
import anndata as ad
from anndata import AnnData
import torch
from sklearn.metrics import accuracy_score,adjusted_rand_score,f1_score,precision_score,recall_score
# from mars.mars import MARS
# from mars.experiment_dataset import ExperimentDataset
# from args_parser import get_parser

# Setting parameters
predictive_alg = "lightGBM"
predictive_alg_rejection = "GaussianNB"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 5
threshold_rejection = 0.8
filter_proportion = 5
data_name = "bench"

# Data preprocessing
# Note: change the path to dataset
adata_raw = sc.read_h5ad('/data/hoan/project24/SemiSupervisedLearning/Data/bench/cellbench.h5ad')
adata_raw.layers['counts'] = adata_raw.X
adata_raw

label_key = 'ground_truth'
batch_key = 'experiment'
adata = adata_raw.copy()


# In[23]:


from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np

def classification_rejection_v2(X, Y, Y_all_labels, X_train, Y_train,X_test,predictive_alg, threshold_rejection):
    if predictive_alg == "svm":
        clf = svm.SVC(probability=True, max_iter=100000).fit(X_train, Y_train)
    # if predictive_alg == "LinearSVM":
    #     svc = LinearSVC()
    #     clf = CalibratedClassifierCV(svc, cv=sss).fit(X_train, Y_train)
        # fix to 5 fold instead of 10 fold ?
    if predictive_alg == "lr":
        clf = LogisticRegression(max_iter=10000).fit(X_train, Y_train)
    if predictive_alg == "lda":
        clf = LinearDiscriminantAnalysis().fit(X_train, Y_train)
    if predictive_alg == "dt":
        clf = DecisionTreeClassifier().fit(X_train, Y_train)
    if predictive_alg == "lightGBM":
        clf = LGBMClassifier(verbose=-1).fit(X_train, Y_train)
    if predictive_alg == "GaussianNB":
        clf = GaussianNB().fit(X_train, Y_train)

    probs_max = [np.max(x) for x in clf.predict_proba(X_test)]
    Y_predict = list(clf.predict(X_test))
    Y_predict_rejection = [Y_predict[i] if probs_max[i] >= threshold_rejection else int(10*len(Y_all_labels) + 1)  for i in range(X_test.shape[0])]
    return  Y_predict_rejection

from scnym2.api import scnym_api
import scanpy as sc
import scnym2
config = scnym2.api.CONFIGS["new_identity_discovery"]
config["n_epochs"] = 10
# increase the weight of the domain adversary 0.1 -> 0.3
config["ssl_kwargs"]["dan_max_weight"] = 0.3
config['batch_size'] = 100

def run_scNym(annotated, unannotated):
    ## scNym
    annotated.X = annotated.X-annotated.X.min()+1.0
    sc.pp.normalize_total(annotated, target_sum=1e6)  # scale each cell to 1e6 total counts
    sc.pp.log1p(annotated)  
    scnym_api(
        adata=annotated,
        task='train',
        groupby='ground_truth',
        out_path='scnym_output',
        config=config
    )
    unannotated.X = unannotated.X-unannotated.X.min()+1
    sc.pp.normalize_total(unannotated, target_sum=1e6)  # scale each cell to 1e6 total counts
    sc.pp.log1p(unannotated)  
    scnym_api(
        adata=unannotated,
        task='predict',
        key_added='scNym',
        trained_model='scnym_output',
        out_path='scnym_output',
        config=config,
    )
    Y_predict_rejection = list(unannotated.obs['scNym'])
    return Y_predict_rejection


# In[24]:


# batch-aware feature selection
sc.pp.highly_variable_genes(adata,n_top_genes=2000, flavor="cell_ranger", batch_key=batch_key)


# use selected genes for integration
adata_hvg = adata[:,adata.var["highly_variable"]].copy()

# run pca
sc.pp.pca(adata_hvg, n_comps=100)
adata_hvg.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

# run combat integration to correct for sample and condition effects
sc.pp.combat(adata_hvg, key=batch_key)


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


# Sample data
exp_10x = adata_hvg[adata_hvg.obs['experiment'] == '10x_5cl',:]
X_exp_10x = exp_10x.X
y_exp_10x = np.array(exp_10x.obs['ground_truth'], dtype=np.int64)

exp_celseq2 = adata_hvg[adata_hvg.obs['experiment'] == 'CelSeq2_5cl',:]
X_celseq2 = exp_celseq2.X
y_celseq2 = np.array(exp_celseq2.obs['ground_truth'], dtype=np.int64)

# Loop to remove label randomly
for i in np.unique(y_exp_10x):
    remove_label = i
    X_10x = X_exp_10x[y_exp_10x != remove_label]
    y_10x = y_exp_10x[y_exp_10x != remove_label]

    print("Data experiment...")
    print(f"Class removed from training:{i}")

    annotated = ad.AnnData(X=X_10x)
    annotated.obs['ground_truth'] = y_10x
    annotated.obs['experiment'] = '10x'

    unannotated = ad.AnnData(X=X_celseq2)
    unannotated.obs['ground_truth'] = y_celseq2
    unannotated.obs['experiment'] = 'celseq2'

    annotated.obs_names = [f"Cell_{i:d}" for i in range(annotated.n_obs)]
    annotated.var_names = [f"Gene_{i:d}" for i in range(annotated.n_vars)]

    unannotated.obs_names = [f"Cell_{i:d}" for i in range(unannotated.n_obs)]
    unannotated.var_names = [f"Gene_{i:d}" for i in range(unannotated.n_vars)]

    annotated_y = np.array(annotated.obs['ground_truth'], dtype=np.int64)
    annotated_x = np.asarray(annotated.X)

    unannotated_y = np.array(unannotated.obs['ground_truth'], dtype=np.int64)
    unannotated_x = np.asarray(unannotated.X)

    
    # # SRNC and Rejection implementation
    data_embbed_x = np.concatenate([annotated_x,unannotated_x])
    data_embbed_y = np.concatenate([annotated_y,unannotated_y])
    y_all_labels = list(set(data_embbed_y))


    Y_predict_rejection=classification_rejection_v2(data_embbed_x,data_embbed_y,y_all_labels,annotated_x,annotated_y,unannotated_x,predictive_alg_rejection,threshold_rejection)
    
	# predictive_alg_rejection = 'scNym'
    Y_predict_scNym=run_scNym(annotated, unannotated)




