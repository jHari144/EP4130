# take the csv acc.csv and it has two columns with no headers, 1st one actual cluster, 2nd one predicted cluster. It has 4 clusters.
# our job is to find the accuracy of the clusters. But we do not know the actual clusters.

# ground truth labels
gtl = [1, 2, 3, 4]
# change these values according to the lables of the clusters
pred_1 = [0, 1, 2, 3]


# calculate the accuracy for each prediction label order

def accuracy(data, gtl, pred):
    i = 0
    for j in range(len(data)):
        if pred[data[j, 0]-1] == data[j, 1]:
            i += 1
    return i/len(data) * 100

import pandas as pd
 
data = pd.read_csv("acc.csv").to_numpy()

print(f"for the assumed prediction order {pred_1}, the accuracy is {accuracy(data, gtl, pred_1)}")