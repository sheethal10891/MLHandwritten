import numpy as np
import csv
from sklearn import cluster
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

## helper.py
def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data


def create_labels(yTe=np.array([[]])):
    n = yTe.shape[0]
    labels = np.zeros((n, ))

    for i in xrange(n):
        labels[i] = np.argmax(yTe[i, :])
    return labels.astype(np.int32)

def generate_upload_file(yTe=np.array([[]]), file_name='temp.csv'):
    if yTe.ndim > 2:
        AssertionError('Either 2d or 1d array')
    elif yTe.ndim == 2:
        yTe = create_labels(yTe)
    n = yTe.shape[0]
    indices = np.array(list(range(6001,10001)))
    print(indices.shape)
    print(yTe.shape)
    df = pd.DataFrame(data={'Id':indices, 'Label':yTe})
    df = df[['Id', 'Label']]
    df.to_csv('./output/'+file_name, header=True, index=False)
    
    
# DataRead and adj_matrix generation

dataset = loadData("Extracted_features.csv")
seed = loadData("Seed.csv", "int")
## Read the Graph nodes data and create an adjacency matrix
filename = "Graph.csv"


adj_matrix = np.zeros((6000, 6000))
print(adj_matrix.shape)

with open(filename, "r") as file:
    reader = csv.reader(file)
    for x,y in reader:
        adj_matrix[int(x)-1,int(y)-1] = 1
        
        
        
## Spectral Clusetering

spectral = cluster.SpectralClustering(n_clusters=10, affinity="precomputed", n_init = 100, assign_labels = "kmeans", n_jobs=-1)

#fit_output = spectral.fit(adj_matrix)

cluster_assign = spectral.fit(adj_matrix)

cluster_assign = cluster_assign.labels_

## Supervised using labels from spectral clustering

from sklearn.decomposition import PCA
from sklearn import datasets, svm
from sklearn import cross_validation
from sklearn import neighbors


#rand_pca = PCA(n_components=870, svd_solver='randomized')
#rand_pca.fit(dataset[:6000])

#print("---------Variance explained-------------------")
#print(rand_pca.explained_variance_ratio_.cumsum())
'''
with plt.style.context('fivethirtyeight'):    
    plt.show()
    plt.xlabel("Principal components ")
    plt.ylabel("Variance")
    plt.plot(rand_pca.explained_variance_ratio_)
    plt.title('Variance Explained by Extracted Componenent')

#plt.show()
'''
train_ext = dataset[:6000]#rand_pca.fit_transform(dataset[:6000])

# New dimensions
print("---------Train-set dimensions after PCA--------")
print(train_ext.shape)


# Check how much time it takes to fit the SVM
start = int(round(time.time() * 1000))


# Fitting training data to SVM classifier.
# Fine-tuning parameters session included
# rbf, poly, linear and different values of gammma and C
classifier= neighbors.KNeighborsClassifier(n_neighbors=10)
#classifier = svm.SVC(gamma=0.01, C=0.1, kernel='rbf')
#classifier.fit(train_ext,y_pred_iso)
classifier.fit(train_ext,cluster_assign)
print(classifier.predict(test_ext))

#print("---------(5) Cross validation accuracy--------")
#print(cross_validation.cross_val_score(classifier, train_ext, y_pred_iso, cv=5))

# End of time benchmark
end = int(round(time.time() * 1000))
print("--SVM fitting finished in ", (end-start), "ms--------------")


# Fitting the new dimensions.
test_ext = dataset[6000:]
#test_ext = rand_pca.transform(dataset[6000:])
print("---------Test-set dimensions after PCA--------")
print(test_ext.shape)
#expected = y_test
predicted = classifier.predict(test_ext)


#generate_upload_file(np.array(cluster_assign))
#ids = seed[:,1]

#print(y_pred_iso[ids])
#print()

print(cluster_assign[1:100])


### Map to correct labels
map_ = {9:0, 0:1, 6:6, 2:5, 3:9, 4:3, 1:2 , 7:7, 8:8, 5:4}
correct_pred = []
for i in range(len(predicted)):
    #print(predicted[i])
    correct_pred.append(map_[int(predicted[i])])
#print(correct_pred)    
correct_pred = np.array((correct_pred))

print(correct_pred.shape)

generate_upload_file(correct_pred, file_name='predictions.csv')
