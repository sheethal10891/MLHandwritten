import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from helper import *
from sklearn.cross_decomposition import CCA

%matplotlib inline

## PCA on feature set
reduced_data = PCA(n_components=870).fit_transform(dataset_norm)


## Spectral Embedding
from sklearn import (manifold, datasets, decomposition)

from sklearn import cluster


# Spectral embedding projection 
print("Computing Spectral embedding")
start = int(round(time.time() * 1000))
X_spec = manifold.SpectralEmbedding(n_components=10, affinity='precomputed', gamma=None, random_state=None, eigen_solver=None).fit_transform(adj_matrix)
end = int(round(time.time() * 1000))
print("--Spectral Embedding finished in ", (end-start), "ms--------------")
print("Done.")

    
cca = CCA(n_components=10)
cca.fit(reduced_data[:6000], X_spec)

Y = cca.transform(reduced_data)

#Y = np.dot(reduced_data,dataset_c)

## Do basic Kmeans

from sklearn.cluster import KMeans
from scipy.stats import mode

kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit(Y)
clusters = clusters.labels_

### Print output for all 10000 data points
for i in range(10):
    print(sum(clusters == i))
    
generate_upload_file(np.array(clusters),'cca_no_pca.csv')


### Map to the clusters in seed and print the output.
map_ = {5:8,1:4,0:2,6:1,7:9,3:5,2:3,4:0,8:6,9:7}

input_ = np.loadtxt('output/cca.csv', delimiter=',',dtype='int', skiprows=1)

preds = []
for id_,lab in input_:
    preds.append(map_[lab])
    
preds = np.array(preds[6000:])

generate_upload_file(preds,'cca_preds.csv')

