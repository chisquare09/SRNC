from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn import svm
from random import choices
from operator import itemgetter
import numpy as np

def EpsilonX2X(X, Y, control_neighbor, filter_proportion):
    Knn_temp = NearestNeighbors(n_neighbors = control_neighbor + 1).fit(X)
    distancesNeighbours = Knn_temp.kneighbors(X)
    distances = distancesNeighbours[0]
    extrated_distances = [np.max(x) for x in distances]
    neighbours = [list(x) for x in distancesNeighbours[1]]
    neigbours_labels = [[Y[i] for i in ind] for ind in neighbours]
    corrected_pairs = [1 if np.min(neigbours_labels[i]) == np.max(neigbours_labels[i]) else 0 for i in range(Y.shape[0])]
    extrated_distances_false = [extrated_distances[i] for i in range(Y.shape[0]) if corrected_pairs[i] ==0]
    if len(extrated_distances_false) > 0:
        return np.percentile(extrated_distances_false, filter_proportion)
    else:
        extrated_distances_true = [extrated_distances[i] for i in range(Y.shape[0]) if corrected_pairs[i] ==1]
        return np.max(extrated_distances_true)

def NeighborCheck_X2X(X, Y, train_indices, comp, control_neighbor, filter_proportion):
    X_temp = PCA(n_components=comp).fit_transform(X)
    clf = NearestNeighbors(n_neighbors = control_neighbor+1).fit(X_temp[train_indices])
    distance_neigbours = clf.kneighbors(X_temp[train_indices])
    neigbours = distance_neigbours[1]
    neigbours = [list(x) for x in neigbours]
    neigbours_labels = [[Y[train_indices[i]] for i in ind] for ind in neigbours]
    corrected_pairs = [1 if np.min(neigbours_labels[i]) == np.max(neigbours_labels[i]) else 0 for i in range(len(train_indices))]
    proprotion_corrected_pairs = np.sum(corrected_pairs)/len(Y[train_indices])
    return proprotion_corrected_pairs


def Classifier(Y_all_labels, new_classes, x_test, epsilon_choice, X_train_temp, Y_train_temp,
            predictive_alg, control_neighbor, threshold_rejection):
    clf = RadiusNeighborsClassifier(radius=epsilon_choice, weights='distance').fit(X_train_temp, Y_train_temp)
    predict_set_0 = clf.radius_neighbors(x_test.reshape(1, -1))[1]
    predict_set_0 = list(predict_set_0[0])
    predict_set_size = len(predict_set_0)
    if predict_set_size > 0:
        X_train_radius, Y_train_radius = X_train_temp[predict_set_0], Y_train_temp[predict_set_0]
        if np.min(Y_train_radius) == np.max(Y_train_radius):
            y_predict = Y_train_radius[0]
        else:
            if predictive_alg == "svm":
                clf = svm.SVC(probability=True, max_iter=10000).fit(X_train_radius, Y_train_radius)
            if predictive_alg == "LinearSVM":
                clf = LinearSVC(max_iter=1000000).fit(X_train_radius, Y_train_radius)

            if predictive_alg=="lightGBM":
                clf = LGBMClassifier(verbose=-1).fit(X_train_radius, Y_train_radius)
            if predictive_alg=='LinearSVM':
                proba_predict = clf._predict_proba_lr(x_test.reshape(1, -1))
                if np.max(proba_predict) < threshold_rejection:
                    y_predict = new_classes
                else:
                    y_predict = clf.predict(x_test.reshape(1, -1))[0]
            else:
                proba_predict = clf.predict_proba(x_test.reshape(1,-1))
                if np.max(proba_predict) < threshold_rejection:
                    y_predict = new_classes
                else:
                    y_predict = clf.predict(x_test.reshape(1, -1))[0]

    else:
        y_predict = new_classes
    return y_predict

#%% Algorithm 1
def SequentialRadiusNeighborsClassifier(X_embedded, Y_all_labels, X_train, X_test, Y_train, predictive_alg,
                                        control_neighbor, shrink_parameter, filter_proportion, threshold_rejection):
    clf = svm.SVC(probability=True, max_iter=10000).fit(X_train, Y_train)
    X_train_temp =  np.copy(X_train)
    Y_train_temp =  np.copy(Y_train)
    test_size = X_test.shape[0]
    Y_predict = [-1 for x in range(test_size)]
    new_classes = int(10*len(Y_all_labels) + 1)
    Knn_temp = NearestNeighbors(n_neighbors= 2).fit(X_embedded)
    max_distances_test = Knn_temp.kneighbors(X_test)[0]
    max_distances_test = [np.max(x) for x in max_distances_test]
    sort_indices = np.argsort(max_distances_test)
    epsilon_choice = EpsilonX2X(X_train, Y_train, control_neighbor, filter_proportion)
    for test_time in range(test_size):
        chosen_test = sort_indices[test_time]
        y_predict = Classifier(Y_all_labels, new_classes, X_test[chosen_test],
                                epsilon_choice, X_train_temp, Y_train_temp, predictive_alg,
                                control_neighbor, threshold_rejection)
        if np.max(clf.predict_proba(X_test[chosen_test].reshape(1, -1))) < threshold_rejection:
                y_predict = Classifier(Y_all_labels, new_classes, X_test[chosen_test],
                                                epsilon_choice, X_train_temp, Y_train_temp, predictive_alg,
                                                control_neighbor, threshold_rejection)
        else:
                y_predict = clf.predict(X_test[chosen_test].reshape(1, -1))[0]
        X_train_temp = np.append(X_train_temp, [X_test[chosen_test]], axis =0)
        Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
        Y_predict[chosen_test] = y_predict
    return Y_predict

#%% Algorithm 2
#%%
def SplitData(X, Y, Y_all_labels,  proportion_unknown, left_out_proportion, random_seed):
    labels_set = list(set(Y))
    labels_counts = [len(list(np.where(Y == y_label)[0])) for y_label in labels_set]
    indices, L_sorted = zip(*sorted(enumerate(labels_counts), key=itemgetter(1)))
    Y_minor_labels = indices[:int(proportion_unknown*len(labels_set))]
    unknown_classes = list(set(choices(Y_minor_labels, k=int(len(Y_minor_labels)))))
    ids = [i for i in range(X.shape[0]) if Y[i] not in unknown_classes]
    if left_out_proportion > 0:
        train_indices, test_indices = train_test_split(ids, test_size = left_out_proportion, random_state = random_seed)
    else:
        train_indices = ids
    test_indices = [i for i in range(X.shape[0]) if i not in train_indices]
    known_classes = list(set(Y[train_indices]))
    actual_unknown_classes = [i for i in labels_set if i not in known_classes]
    print("information:", "\n classes: ", listToStringWithoutBrackets(Y_all_labels), "\n unknown_classes: ", listToStringWithoutBrackets(actual_unknown_classes))
    return [train_indices, test_indices, actual_unknown_classes]

def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')

def matchLabel(Y_labels, Y_ref):
  # this function changes clusters' label of Y_labels s.t ARI(Y_labels, Y_ref) is maximal.
  Y_labels_set = np.unique(Y_labels)
  Y_ref_set = np.unique(Y_ref)
  Y_result = np.copy(Y_labels)
  for x in Y_labels_set:
      arg_index = -1
      max_value = -1
      for y in Y_ref_set:
          setA = (Y_labels == x)
          setB = (Y_ref == y)
          sumAB = np.sum(setA*setB)
          if sumAB > max_value:
              max_value = sumAB
              arg_index = y
      Y_result[Y_labels==x] = arg_index #replace x in Y_labels by arg_index
  return Y_result


