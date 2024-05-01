import argparse
from time import perf_counter, process_time, time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.random import uniform


def k_means(data, k, max_iter=100, epsilon = 0.01, verbose = False):
    np.random.seed()
    if verbose:
        print("ITERATION; points per cluster")

    dims = data.shape[1]
    data_points = data.shape[0]
    dim_min = data.min(axis=0)
    dim_max = data.max(axis=0)
    dim_avg = (dim_min + dim_max)/2

    while True:
        means = uniform(dim_min,dim_max,(k,dims))
        clusters = np.zeros(data_points,dtype=int)
        for i in range(data_points):
            clusters[i] = norm((data[i]-means)/dim_avg,axis=1).argmin()
        if np.prod([(clusters==i).sum() for i in range(k)]) != 0:
            break


    for iter in range(max_iter):
        for i in range(data_points):
            clusters[i] = norm((data[i]-means)/dim_avg,axis=1).argmin()
        prev_means = means.copy()
        means = np.array([data[clusters==i].mean(axis=0) for i in range(k)])
        if verbose:
            print(f"{iter}. {[(clusters==i).sum() for i in range(k)]}")
        if norm((prev_means-means)/dim_avg) < epsilon:
            break

    return means, clusters, iter+1

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(usage="k-means.py [--help] data K [--max_iter MAX_ITER] [--epsilon EPSILON] [--output_clusters OUTPUT_CLUSTERS] [--output_means OUTPUT_MEANS] [--verbose] [--benchmark] [--PCA]",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("data",
                           type = str,
                           help = "Path to the input csv data file (data should have shape - (samples, features))")
    argparser.add_argument("K",
                           type = int,
                           help = "Number of clusters to be formed")
    argparser.add_argument("--max_iter",
                           "-m",
                            type = int,
                            default = 100,
                            help = "Maximum number of iterations for convergence")
    argparser.add_argument("--epsilon",
                            "-e",
                            type = float,
                            default = 0.01,
                            help = "Convergence threshold")
    argparser.add_argument("--output_clusters",
                            "-oc",
                            type = str,
                            default = "./clusters.csv",
                            help = "Path to the output clusters file")
    argparser.add_argument("--output_means",
                            "-om",
                            type = str,
                            default = "./means.csv",
                            help = "Path to the output means file")
    argparser.add_argument("--verbose",
                            "-v",
                            action = "store_true",
                            help = "Print the progress of the algorithm")
    argparser.add_argument("--benchmark",
                            "-b",
                            action = "store_true",
                            help = "Print benchmarking information")
    argparser.add_argument("--PCA",
                            "-p",
                            action = "store_true",
                            help = "Applying PCA")
    
    args = argparser.parse_args()

    data = pd.read_csv(args.data,header=None).to_numpy(dtype=float)

    if args.benchmark:
        tic1 = time()
        tic2 = perf_counter()
        tic3 = process_time()

    # means, clusters, iters = k_means(data, args.K, args.max_iter, args.epsilon, args.verbose)

    if not args.PCA:
        means, clusters, iters = k_means(data, args.K, args.max_iter, args.epsilon, args.verbose)
    else:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        print("Initial data shape: ",data.shape)
        # plot a 3d scatter plot of the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0],data[:,1],data[:,2],s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Data points')
        plt.show()
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        print("PCA data shape: ",data.shape)
        # plot the PCA data
        plt.scatter(data[:,0],data[:,1],s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PCA Data points')
        plt.show()
        means, clusters, iters = k_means(data, args.K, args.max_iter, args.epsilon, args.verbose)
        # store the new PCA data in PCA_<data_path>.csv
        pd.DataFrame(data).to_csv("PCA_"+args.data,header=None,index=None)

    if args.benchmark:
        toc1 = time()
        toc2 = perf_counter()
        toc3 = process_time()

        print("\nBenchmarks:--")
        print("Data size: ",data.shape)
        print(f"K = {args.K}")
        print(f"epsilon = {args.epsilon}")
        print(f"iterations = {iters}")
        print(f"time: {toc1-tic1:.3f} seconds")
        print(f"perf_counter : {toc2-tic2:.3f} seconds")
        print(f"process_time : {toc3-tic3:.3f} seconds\n")
        

    pd.DataFrame(means).to_csv(args.output_means,header=None,index=None)
    pd.DataFrame(clusters).to_csv(args.output_clusters,header=None,index=None)
