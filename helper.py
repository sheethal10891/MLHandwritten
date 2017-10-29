import numpy as np
import pandas as pd

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
    indices = np.array(list(range(n)))
    df = pd.DataFrame(data={'id':indices, 'digit':yTe})
    df = df[['id', 'digit']]
    df.to_csv('./output/'+file_name, header=True, index=False)
