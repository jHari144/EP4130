# take the data from the csv file and plot it. plot 1000 random data points from the file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_data_cluster(data_path, clusters_data, x_label, y_label, dimensions):
    data = pd.read_csv(data_path).to_numpy()
    clusters = pd.read_csv(clusters_data).to_numpy()
    # X_labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    # X_labels = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    # scatter plot the datapoints from the data where cluster data has the same cluster number
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    if dimensions == 2:
        # plt.figure(figsize=(9, 10))
        # plt.subplot(321)
        for i in range(0, len(np.unique(clusters))):
            data_points = data[clusters[:,0]==i]
            plt.scatter(data_points[:,0], data_points[:,1], s=1, c=colors[i-1], label=f'{i}')
            # plt.xlabel(X_labels[0]) # 'X
            # plt.ylabel(X_labels[1]) # 'Y
            # # plt.title('Data points')
        #     plt.legend()
        # plt.subplot(322)
        # for i in range(0, len(np.unique(clusters))):
        #     data_points = data[clusters[:,0]==i]
        #     plt.scatter(data_points[:,0], data_points[:,2], marker='o', c=colors[i-1], label=f'{i}')
        #     plt.xlabel(X_labels[0]) # 'X
        #     plt.ylabel(X_labels[2]) # 'Y
        #     # plt.title('Data points')
        #     plt.legend()
        # plt.subplot(323)
        # for i in range(0, len(np.unique(clusters))):
        #     data_points = data[clusters[:,0]==i]
        #     plt.scatter(data_points[:,0], data_points[:,3], marker='o', c=colors[i-1], label=f'{i}')
        #     plt.xlabel(X_labels[0]) # 'X
        #     plt.ylabel(X_labels[3]) # 'Y
        #     # plt.title('Data points')
        #     plt.legend()
        # plt.subplot(324)
        # for i in range(0, len(np.unique(clusters))):
        #     data_points = data[clusters[:,0]==i]
        #     plt.scatter(data_points[:,1], data_points[:,2], marker='o', c=colors[i-1], label=f'{i}')
        #     plt.xlabel(X_labels[1]) # 'X
        #     plt.ylabel(X_labels[2]) # 'Y
        #     # plt.title('Data points')
        #     plt.legend()
        # plt.subplot(325)
        # for i in range(0, len(np.unique(clusters))):
        #     data_points = data[clusters[:,0]==i]
        #     plt.scatter(data_points[:,1], data_points[:,3], marker='o', c=colors[i-1], label=f'{i}')
        #     plt.xlabel(X_labels[1]) # 'X
        #     plt.ylabel(X_labels[3]) # 'Y
        #     # plt.title('Data points')
        #     plt.legend()
        # plt.subplot(326)
        # for i in range(0, len(np.unique(clusters))):
        #     data_points = data[clusters[:,0]==i]
        #     plt.scatter(data_points[:,2], data_points[:,3], marker='o', c=colors[i-1], label=f'{i}')
        #     # plt.scatter(data_points[:,2], data_points[:,1], label=f'Cluster {i}', s=1)
        #     plt.xlabel(X_labels[2]) # 'X
        #     plt.ylabel(X_labels[3]) # 'Y
        #     # plt.title('Data points')
        #     plt.legend()
        plt.title('K-means Clustering of 2D Syntetic Data')
        # plt.tight_layout()
        plt.show()
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, len(np.unique(clusters))):
            data_points = data[clusters[:,0]==i]
            # ax.scatter(data_points[:,0], data_points[:,1], data_points[:,2], c=colors[i-1], label=f'Cluster {i}', s=1)
            ax.scatter(data_points[:,0], data_points[:,1], data_points[:,2], label=f'{i}', s=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('Z')
        ax.set_title('K-means Clustering of 3D Syntetic Data')
        plt.legend()
        plt.show()
    elif dimensions == 1:
        unique_labels = ["Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        # plt.figure(figsize=(, 11))
        for i, label in enumerate(unique_labels):
            plt.subplot(4,3,i+1)
            for j in range(0, len(np.unique(clusters))):
                data_points = data[clusters[:,0]==j]
                plt.scatter(data_points[:,i], np.zeros_like(data_points[:,i]), s=5, c=colors[j-1], label=f'{j}')
            plt.xlabel(label)
            plt.ylabel('Y')
            # plt.legend()
        plt.suptitle('K-means Clustering of 1D Syntetic Data with 3 Clusters')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # data_path = "accidents.csv"
    # data_path = "housing.csv"
    # data_path = "generated.csv"
    # clusters_data = "clusters.csv"
    argparser = argparse.ArgumentParser(usage="plotter.py [--help] data clusters [--x_label x_label] [--y_label y_label] [--dimensions dim]",)
    # argparser = argparse.ArgumentParser()#usage="plotter.py [--help] data clusters [--x_label x_label] [--y_label y_label]",)
    argparser.add_argument("data",
                           type = str,
                           help = "Path to the input csv data file (data should have shape - (samples, features))")
    argparser.add_argument("clusters",
                            type = str,
                            help = "Path to the input csv clusters file (clusters should have shape - (samples, 1))")
    argparser.add_argument("--x_label", "-xl",
                            type = str,
                            default = 'X',
                            help = "X label")
    argparser.add_argument("--y_label", "-yl",
                            type = str,
                            default = 'Y',
                            help = "Y label")
    argparser.add_argument("--dimensions", "-di",
                            type = int,
                            default = 2,
                            help = "Number of dimensions in the data")
    args = argparser.parse_args()

    data_path = args.data
    clusters_data = args.clusters
    
    plot_data_cluster(data_path, clusters_data, args.x_label, args.y_label, args.dimensions)