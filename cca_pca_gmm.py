import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
%matplotlib inline
from sklearn import (manifold, datasets, decomposition)
from sklearn.cross_decomposition import CCA
from sklearn.mixture import GaussianMixture
from sklearn import cluster



mean_ds = []
for i in range(dataset.shape[1]):
    mean_ds.append(np.mean(dataset[:,i]))
mean_ds = np.array(mean_ds)

dataset_norm = dataset - mean_ds

### CCA



## PCA on feature set
reduced_data = PCA(n_components=51).fit_transform(dataset_norm)


## Spectral Embedding


# Spectral embedding projection 
print("Computing Spectral embedding")
start = int(round(time.time() * 1000))
X_spec = manifold.SpectralEmbedding(n_components=151, affinity='precomputed', gamma=None, random_state=None, eigen_solver=None).fit_transform(adj_matrix)
end = int(round(time.time() * 1000))
print("--Spectral Embedding finished in ", (end-start), "ms--------------")
print("Done.")



    
cca = CCA(n_components=22)
cca.fit(dataset_norm[:6000], X_spec)

Y = cca.transform(dataset_norm)

#Y = np.dot(reduced_data,dataset_c)


### Do GMM
gmm = GaussianMixture(n_components=10).fit(Y)
clusters = gmm.predict(Y)

## Output of GMM for 10,000 to a csv file
for i in range(10):
    print(sum(clusters == i))
    
generate_upload_file(np.array(clusters),'cca_pca_gmm.csv')


### After Manual Mapping

map_ = {5:3,1:8,0:5,6:4,7:9,3:0,2:2,4:1,8:6,9:7}

input_ = np.loadtxt('output/cca_pca_gmm.csv', delimiter=',',dtype='int', skiprows=1)

preds = []
for id_,lab in input_:
    preds.append(map_[lab])
    
preds = np.array(preds[6000:])

generate_upload_file(preds,'cca_pca_gmm_preds.csv')