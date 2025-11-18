#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import torch
from sklearn.metrics import accuracy_score,adjusted_rand_score,f1_score,precision_score,recall_score
# from mars.mars import MARS
# from mars.experiment_dataset import ExperimentDataset
# from args_parser import get_parser
from sklearn.decomposition import PCA

# Setting parameters
predictive_alg = "lightGBM"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 1 
threshold_rejection = 0.3 
filter_proportion = 0 
data_name = "PMBCs"

# Note: change the path to dataset
reference = pd.read_csv('/data/hoan/project24/SemiSupervisedLearning/Data/PMBCs/reference.csv', delimiter=',', header=None)
query = pd.read_csv('/data/hoan/project24/SemiSupervisedLearning/Data/PMBCs/query.csv', delimiter=',', header=None)

X_reference = reference.iloc[:,1:].to_numpy(dtype=float)
y_reference = reference.iloc[:, 0].to_numpy(dtype=int)
X_query = query.iloc[:,1:].to_numpy(dtype=float)
y_query = query.iloc[:, 0].to_numpy(dtype=int)

# Data preprocessing: PCA
pca = PCA(n_components=100)
X_reference = pca.fit_transform(X_reference)
X_query = pca.transform(X_query)


# In[ ]:


from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import anndata as ad

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
config["n_epochs"] = 50
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


# In[11]:


predictive_alg_rejection = "lr"

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


# Loop to remove label randomly
for i in np.unique(y_reference):
    remove_label = i
    X_query = X_reference[y_reference != remove_label]
    y_query = y_reference[y_reference != remove_label].astype(dtype=int)

    num_cells_query, num_genes_query = X_query.shape
    num_cells_query, num_genes_query = X_query.shape

    print("Data experiment...")
    print(f"Class removed from training:{i}")

    annotated = ad.AnnData(X=X_reference)
    annotated.obs['ground_truth'] = y_reference

    unannotated = ad.AnnData(X=X_query)
    unannotated.obs['ground_truth'] = y_query

   

    # SRNC and Rejection implementation
    annotated_x = X_query
    annotated_y = y_query
    unannotated_x = X_query
    unannotated_y = y_query
    data_embbed_x=np.concatenate([annotated_x,unannotated_x])
    data_embbed_y=np.concatenate([annotated_y,unannotated_y])

    y_all_labels = list(set(data_embbed_y))

    # Y_predict_srnc=SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,
    #                                         control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)


    Y_predict_rejection=classification_rejection_v2(data_embbed_x,data_embbed_y,y_all_labels,annotated_x,annotated_y,unannotated_x,predictive_alg_rejection,threshold_rejection)
    
    # predictive_alg_rejection = 'scNym'
    Y_predict_scNym=run_scNym(annotated, unannotated)

 




