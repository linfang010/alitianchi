# filename: plot.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler



def plot_pca_3d():

    data = pd.read_csv('clean_train_data.csv')
    X = data.ix[:,1:45]
    y = data['label']
    
    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig,rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    
    
    X = StandardScaler().fit_transform(X)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    print (len(X))

    
    for name,label in [('negtive', 0), ('positive', 1)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    
if __name__ == '__main__':
    
    plot_pca_3d()
                

    
    
