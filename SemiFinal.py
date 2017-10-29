import numpy as np
import matplotlib.pyplot as plt

from helper import *
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation

from sklearn.metrics import confusion_matrix, classification_report

X = loaddata('./data/Extracted_features.csv')
labeledData=loaddata('./data/Seed.csv',data_type='int')
indices = np.arange(10000)
y=np.zeros((10000,1))
y[indices]=-1

y_train=np.copy(y)
labeledData=labeledData.reshape((60,2))
lindices=labeledData[:,0]#.reshape(60,1)
y_train[lindices-1]= labeledData[:,1].reshape(60,1)


lp_model = label_propagation.LabelSpreading( max_iter=1)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[indices]

print predicted_labels[9999]
print predicted_labels[5999:10000].shape

generate_upload_file(predicted_labels[5999:10000])
