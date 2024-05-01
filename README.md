# Usage

## `k-means.py`
```
usage: k-means.py [--help] data K [--max_iter MAX_ITER] [--epsilon EPSILON]
                  [--output_clusters OUTPUT_CLUSTERS] [--output_means OUTPUT_MEANS] [--verbose] [--benchmark] [--PCA]

positional arguments:
  data                  Path to the input csv data file (data should have
                        shape - (samples, features))
  K                     Number of clusters to be formed

options:
  -h, --help            show this help message and exit
  --max_iter MAX_ITER, -m MAX_ITER
                        Maximum number of iterations for convergence (default:
                        100)
  --epsilon EPSILON, -e EPSILON
                        Convergence threshold (default: 0.01)
  --output_clusters OUTPUT_CLUSTERS, -oc OUTPUT_CLUSTERS
                        Path to the output clusters file (default:
                        ./clusters.csv)
  --output_means OUTPUT_MEANS, -om OUTPUT_MEANS
                        Path to the output means file (default: ./means.csv)
  --verbose, -v         Print the progress of the algorithm (default: False)
  --benchmark, -b       Print benchmarking information (default: False)
  --PCA, -p             Applying PCA (default: False)
```

## `plotter.py`
```
usage: plotter.py [--help] data clusters [--x_label x_label] [--y_label y_label] [--dimensions dim]

positional arguments:
  data                  Path to the input csv data file (data should have
                        shape - (samples, features))
  clusters              Path to the input csv clusters file (clusters should
                        have shape - (samples, 1))

options:
  -h, --help            show this help message and exit
  --x_label X_LABEL, -xl X_LABEL
                        X label
  --y_label Y_LABEL, -yl Y_LABEL
                        Y label
  --dimensions DIMENSIONS, -di DIMENSIONS
                        Number of dimensions in the data
```