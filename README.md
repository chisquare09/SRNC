# SRNC
Semi-supervised learning for Robust Novel Cell-type identification (SRNC) is a novel semi-supervised framework that enhances classification accuracy while effectively identifying unknown cell types. By integrating self-supervised feature learning with semi-supervised classification, SRNC leverages both labeled and unlabeled data to improve generalization. 
# Setup

SRNC require some libraries. Please check the requirements.txt file for more information. You can create new environment and install all required packages with: 

```
pip install -r requirements.txt
```

This code was tested on Python 3.8
# Dataset

The dataset for each experiments can be found at: 
https://drive.google.com/drive/folders/1BvQRTmY3cZYYQzUTFkuyeZvg6PFppq1D?usp=sharing


# Using SRNC
We implemented SRNC model as SequentialRadiusNeighborsClassifier function. To training SRNC: 

```
from model.srnc import SequentialRadiusNeighborsClassifier

Y_predict_srnc=SequentialRadiusNeighborsClassifier(X_embedded, y_all_labels, X_train, X_test, Y_train, predictive_alg,control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)

```

This function require the following input: 
* X_embedded: a lower dimensional representation of the original dataset, which is the embedded feature space of dataset.
* Y_all_labels: a list of all possible class labels in the datasets.
* X_train: feature matrix for the training set.
* X_test: feature matrix for the test set.
* Y_train: ground truth labels for X_train.
* predictive_alg: the classification algorithm used (in this case we used lightGBM).
* control_neighbor: the number of neareast neighbors considered for certain calculations.
* shrink_parameter: a parameter controlling the shrinking process in the classification.
* filter_proportion: proportion of training samples used for certain selection steps.
* threshold_rejection: a probability threshold for rejecting uncertain predictions.

The output of this function is a list of predicted class labels ```Y_predict``` for each test sample in ```X_test```

# Evaluation
To evaluate the model, we can extract the predicted lable ```Y_predict_srnc``` from SRNC model to compare with the ```Y_test``` label from the test dataset using sevaral built-in function provided by ```sklearn.metrics```. For instance: 
```
from sklearn.metrics import adjusted_rand_score, accuracy

adj_score = adjusted_rand_score(Y_predict_srnc,Y_test)
accuracy_score = accuracy(Y_predict_srnc, Y_test)
```
# Reproducibility

The Python scripts to reproduce the results in our paper for each section are included in this directory (experiment1, ecperiment2, experiment3). For each experiment, please run the corresponding .py file.






