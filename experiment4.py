#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Py310


# ## Load libraries and functions

# In[2]:


from model.srnc import SequentialRadiusNeighborsClassifier
# from model.rejection import classification_rejection_v2

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
# from mars.mars import MARS
# from mars.experiment_dataset import ExperimentDataset
# from args_parser import get_parser


# In[3]:


from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np

def classification_rejection_v2(X, Y, Y_all_labels, X_train, Y_train,X_test,predictive_alg, threshold_rejection,sss):
    if predictive_alg == "svm":
        clf = svm.SVC(probability=True, max_iter=100000).fit(X_train, Y_train)
    if predictive_alg == "LinearSVM":
        svc = LinearSVC()
        clf = CalibratedClassifierCV(svc, cv=sss).fit(X_train, Y_train)
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
config['batch_size'] = 50

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


# ## Run methods

# In[4]:


# Setting parameters
predictive_alg = "lightGBM"
predictive_alg_rejection = "lda" # "GaussianNB" "lr"
embedded_option = "PCA"
shrink_parameter = 1
proportion_unknown = 0.2
control_neighbor = 5 
threshold_rejection = 0.3 
filter_proportion = 0 
data_set = ['pollen', 'patel', 'muraro', 'xin', 'zeisel', 'baron']
# data_set = ['patel']
# data_name = 'test'


import numpy as np
import scanpy as sc
from sklearn.decomposition import FastICA, NMF
from sklearn.manifold import MDS

# Optional: scvi needs to be installed: pip install scvi-tools
import scvi

def run_dimensionality_reduction(X, method="ICA", n_components=10, adata=None):
    """
    Perform dimensionality reduction using ICA, MDS, NMF, or scVI.
    
    Parameters
    ----------
    X : np.ndarray or AnnData
        Expression matrix (cells × genes). If AnnData is provided, expression data
        will be taken from adata.X automatically.
    method : str
        One of {"ICA", "MDS", "NMF", "scVI"}.
    n_components : int
        Number of reduced dimensions.
    adata : AnnData, optional
        Required only for scVI (for model setup and storage).

    Returns
    -------
    embedding : np.ndarray
        Low-dimensional representation of shape (n_cells, n_components).
    """
    
    # Handle AnnData input
    if isinstance(X, sc.AnnData):
        adata = X
        X = adata.X

    method = method.upper()

    if method == "ICA":
        model = FastICA(n_components=n_components, random_state=42)
        embedding = model.fit_transform(X)

    elif method == "MDS":
        model = MDS(n_components=n_components, random_state=42, n_jobs=-1)
        embedding = model.fit_transform(X)

    elif method == "NMF":
        model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
        embedding = model.fit_transform(np.abs(X))  # NMF requires nonnegative input

    elif method == "SCVI":
        if adata is None:
            adata = ad.AnnData(X-X.min()+1)
            # raise ValueError("scVI requires an AnnData object (adata parameter).")
        # scVI setup and model training
        scvi.model.SCVI.setup_anndata(adata)
        model = scvi.model.SCVI(adata, n_latent=n_components)
        model.train()
        embedding = model.get_latent_representation()

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'ICA', 'MDS', 'NMF', 'scVI'.")

    return embedding


# In[7]:


# loop through datasets
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
    dimreduce_all = []
    for dimethod in ["ICA", "MDS", "NMF", "SCVI"]:
    # for dimethod in ["SCVI"]:
        if data_name=='muraro':
            k_fold = 3
        else:
            k_fold = 5
        data = pd.read_csv('/data/hoan/project24/SemiSupervisedLearning/Data/'+data_name+'-prepare-log_count_100pca.csv', header=None)

        data_new = run_dimensionality_reduction(X=data.iloc[:, 1:], method=dimethod, n_components=50)
        data = pd.DataFrame(np.concatenate([data.iloc[:, 0].to_numpy().reshape(-1, 1), data_new], axis=1))
    
        # Data sampling: train and test set
        data = pd.DataFrame(data)
        annotated_data = pd.DataFrame()
        unannotated_data = pd.DataFrame()
        
        
        sss = StratifiedShuffleSplit(n_splits=k_fold,train_size=0.8,random_state=0)
        for train_index, test_index in sss.split(data.iloc[:,1:],data.iloc[:,0]):
            annotated_data = data.iloc[train_index]
            unannotated_data = data.iloc[test_index]
        
            # remove randomly 1 labels
            # annotated_data = annotated_data[annotated_data[0] != np.random.choice(annotated_data.iloc[:,0].unique())]
            annotated_data = annotated_data[annotated_data.iloc[:, 0] != np.random.choice(annotated_data.iloc[:, 0].unique())]
        
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
            # annotated_set = ExperimentDataset(annotated_x, annotated.obs_names, annotated.var_names, data_name, annotated_y)
        
            unannotated_y = np.array(unannotated.obs['ground_truth'], dtype=np.int64)
            unannotated_x = np.asarray(unannotated.X)
            # unannotated_set = ExperimentDataset(unannotated_x, unannotated.obs_names, unannotated.var_names, data_name, unannotated_y)
            # pretrain_data = ExperimentDataset(unannotated_x, unannotated.obs_names, unannotated.var_names, data_name)
        
            n_clusters = len(np.unique(unannotated_data.iloc[0,:]))
            
            # SRNC and Rejection implementation
            data_embbed_x=np.concatenate([annotated_x,unannotated_x])
            data_embbed_y=np.concatenate([annotated_y,unannotated_y])
            y_all_labels = list(set(data_embbed_y))
        
            Y_predict_srnc=SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,
                                                    control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)
            



