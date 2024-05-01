# generate three clusters of data points and plot them. each one with a differnt mean and variance. store the data in a csv file with no header and has the name generated.csv

import numpy as np
import pandas as pd

def generate_data():
    # np.random.seed()
    # generate three random means and variances
    means = np.random.rand(3,3)*25
    # variances = np.random.rand(3,2)*3
    variances = [1.8, 3.3, 2.5]
    print(means)
    print(variances)

    data = np.zeros((15000,3))
    for i in range(3):
        data[i*5000:(i+1)*5000] = np.random.normal(means[i],variances[i],(5000,3))
    


    np.random.shuffle(data)
    pd.DataFrame(data).to_csv("generated_.csv",header=False,index=False)

    # plot the data
    import matplotlib.pyplot as plt
    plt.scatter(data[:,0],data[:,1],s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data points')
    plt.show()

if __name__ == "__main__":
    generate_data()