# Author: Matthew Rossi

import numpy as np
from sklearn.utils import resample
import scipy.sparse as sps
        
def balance_classes(X, y, max_class_size=None):
   
    cls_list = np.unique(y)
    if max_class_size == None:
        max_class_size = np.max(np.bincount(y))
    else:
        max_class_size = min(max_class_size, np.max(np.bincount(y)))
    len_after_res = len(cls_list) * max_class_size

    sparse = sps.issparse(X)
    if sparse:
        dim = X.shape[1]
        X_res = sps.csr_matrix([],shape=(1,dim))
    else:
        X = np.array(X)
        X_shape = list(X.shape)
        X_shape[0] = len_after_res
        X_res = np.zeros(X_shape,dtype='int64')
    y_res = np.zeros(len_after_res)
    y = np.array(np.ravel(y), dtype='int64')
    
    for clsnum, cls in enumerate(cls_list):
        X_cls_res, y_cls_res = resample(X[y==cls], y[y==cls], n_samples=max_class_size, random_state=0)
        if sparse:
            X_res = sps.vstack((X_res, X_cls_res))
        else:
            X_res[clsnum*max_class_size:(clsnum+1)*max_class_size] = X_cls_res
        y_res[clsnum*max_class_size:(clsnum+1)*max_class_size] = y_cls_res

    print("The dataset has %i classes upsampled to %i datapoints." % (len(cls_list), max_class_size)) 
    if sparse:
        return X_res[1:], np.array(y_res, dtype='int64')
    else:
        return X_res, np.array(y_res, dtype='int64')
