#!/usr/bin/python

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def main():
    sample_volume = int(input("Enter the desired size of the sample for blobbery:    "))
    sample_feats = int(input("Enter the desired number of features:    "))
    sample_centers = int(input("Enter the desired number of centres:    "))
    sample_std = float(input("Enter the std of the clusters:    "))

    data = make_blobs(n_samples = sample_volume, n_features = sample_feats, centers = sample_centers, cluster_std = sample_std, random_state = 101) 

    plt.scatter(data[0][:,0], data[0][:,1], c = data[1])

    kmeans_train(data,sample_centers)

    plt.show()

def kmeans_train(data,clusters):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(data[0])

    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

    f, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))
    ax1.set_title('K Means')
    ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_)

    ax2.set_title('Original')
    ax2.scatter(data[0][:,0],data[0][:,1],c=data[1])

if __name__ == "__main__": main()
