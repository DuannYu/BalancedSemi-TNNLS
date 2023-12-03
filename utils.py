from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import h5py
import math
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def LoadData(data_name):
    if data_name in ['mnist','fmnist']:
        f = np.load('./data/'+data_name+'.npz')
        x = np.concatenate((f['x_train'], f['x_test']))
        y = np.concatenate((f['y_train'], f['y_test'])).astype(np.int32)
        f.close()
        x = x.reshape((x.shape[0], -1)).astype(np.float32)
        x = np.divide(x, 255.)
        return x, y

    if data_name == 'Reuters':
        data = sio.loadmat('./data/reuters10k.mat')
        # x = data['Feature']
        # y = data['Labels'].squeeze()
        x = data['X']
        y = data['Y'].squeeze()
        return x, y

    if data_name == 'stl-10':
        data = sio.loadmat('./data/stl-10.mat')
        # x = data['Feature']
        # y = data['Labels'].squeeze()
        x = data['data'].astype(np.float32)
        y = data['labels'].squeeze()
        return x, y

    if data_name == 'usps':
        hf = h5py.File('./data/usps.h5', 'r')
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

        x = np.r_[X_tr, X_te]
        y = np.r_[y_tr, y_te]
        return x, y

class LoadDataset(Dataset):

    def __init__(self, dataset_name):
        self.x, self.y = LoadData(dataset_name)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
 
	L1 = y_true - np.min(y_true)+1
	L2 = y_pred - np.min(y_pred)+1
	# compute contingency matrix (also called confusion matrix)
	conti_matrix = contingency_matrix(L1, L2)
	# find optimal one-to-one mapping between cluster labels and true labels
	row_ind, col_ind = linear_sum_assignment(-conti_matrix)
	# return cluster accuracy
	return conti_matrix[row_ind, col_ind].sum() / np.sum(conti_matrix)

# Standard Deviation in Cluster Sizes
def SDCS(y_true, y_pred):

    n = np.max(y_true.shape)
    k = np.unique(y_true)
    c = np.max(k.shape)
    l = np.expand_dims(y_pred, 1) - np.expand_dims(k, 0)
    p = (l == 0).sum(0)
    return (np.sum((p - n/c)**2)/(c-1))**0.5

def NE(y_true, y_pred):

    n = np.max(y_true.shape)
    k = np.unique(y_true)
    c = np.max(k.shape)
    l = np.expand_dims(y_pred, 1) - np.expand_dims(k, 0)
    p = (l == 0).sum(0) + 1e-16
    return -np.sum((p/n)*np.log(p/n))/np.log(c)


