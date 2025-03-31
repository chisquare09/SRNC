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

Y_predict_srnc=SequentialRadiusNeighborsClassifier(data_embbed_x, y_all_labels, annotated_x, unannotated_x, annotated_y, predictive_alg,control_neighbor, shrink_parameter, filter_proportion, threshold_rejection)

```
# Reproducibility

The Python scripts to reproduce the results in our paper for each section are included in this directory (experiment1, ecperiment2, experiment3). For each experiment, please run the corresponding .py file.






