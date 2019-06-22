import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('balance.data')
data = np.array(data)
print(data.shape[0])
label = data[:, 0]
data = data[:, 1:]

for i in range(len(label)):
    if label[i] == 'L':
        label[i] = 0
    elif label[i] == 'R':
        label[i] = 1
    elif label[i] == 'B':
        label[i] = 2

def Normalize(x):
    '''
    Normalization: to 2D data feature
    '''
    num_feature = x.shape[1]
    
    for i in range(num_feature):
        Max = np.max(x[:, i])
        Min = np.min(x[:, i])
        x[:, i] = (x[:, i] - Min)/(Max - Min)
        
    return x

# Do normalization, to make the distance weight of each feature equal
data = Normalize(data)

# Shuffling
permutation = np.random.permutation(data.shape[0])
data = data[permutation]
label = label[permutation]

train_end = int(data.shape[0]*0.8)
x_train = data[0:train_end, :]
y_train = label[0:train_end]
x_test = data[train_end:, :]
y_test = label[train_end:]

print('Data size:', x_train.shape)

def Euclidean(x1, x2):
    '''
    x1, x2 should have same 1D feature length
    '''
    return np.sqrt(sum((x1 - x2)**2))

def distance(x, x_data):
    '''
    calculate the Euclideance distance btw single sample x and all other data sets
    Note that x is not contained in x_data
    x: single data sample
    x_data: whole data set [batch, features]
    return: 1D distance size [batch]
    '''
    dist =[]
    for sample in range(len(x_data)):
        total_dist = 0
        for feature in range(len(x)):
            total_dist += (x[feature] - x_data[sample, feature])**2
            
        total_dist = np.sqrt(total_dist)
        dist.append(total_dist)
    
    dist = np.array(dist)
    return dist

# KNN algorithm
def KNN(x, k, x_data, y_data):
    '''
    KNN algorithm, select the nearest K neighbors to predict the current class of x
    x: single sample
    k: number of nearest selected neighbors
    x_data: features of data sets relied by KNN search
    y_data: label classes of data sets
    return: class of input sample x
    '''
    neighbor_distance = distance(x, x_data)
    neighbor_dist_sort = np.argsort(neighbor_distance) # the sorted distance index 
    k_neighbor_dist = neighbor_dist_sort[0:k]
    k_neighbor_class = y_data[k_neighbor_dist]
    classes, count = np.unique(k_neighbor_class, return_counts = True) # return all neighbor classes and counts
    Max = np.where(count == max(count)) # return the maximum counted class
    out_class = classes[Max]
    
    return out_class

# Start predicting
# 1. Use training data set to predict test data set class
predict = []
for sample in range(len(x_test)):
    predict_class = KNN(x_test[sample], 10, x_train, y_train)
    predict.append(predict_class[0])

predict = np.array(predict)
accuracy = (predict == y_test)
accuracy = np.mean(accuracy)

print('Accuracy:', accuracy)

# Use train data set to predict train data set class
predict = []
for sample in range(len(x_train)):
    search_data = np.delete(x_train, (sample), axis = 0)
    search_label = np.delete(y_train, (sample), axis = 0)
    predict_class = KNN(x_train[sample], 10, search_data, search_label)
    predict.append(predict_class[0])

predict = np.array(predict)
accuracy = (predict == y_train)
accuracy = np.mean(accuracy)

print('Accuracy:', accuracy)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    















