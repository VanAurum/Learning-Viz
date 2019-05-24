"""
Learning Viz : Visualizing the ML learning process
@author: Kevin Vecmanis
"""

#Standard Python library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (StratifiedKFold, RepeatedKFold, KFold, cross_val_score)

import warnings

import random

#3rd party imports

#Local imports
from learning_viz.model_parameters import parameters
from learning_viz.gif_maker import make_gif

warnings.filterwarnings("ignore")



 

    

def plot_decision_bounds(dataset, names, classifiers, file_class, directory=None):
    
    '''
    This function takes in a list of classifier variants and names and plots the decision boundaries
    for each on three different datasets that different decision boundary solutions.
    
    Parameters: 
        names: list, list of names for labelling the subplots.
        classifiers: list, list of classifer variants for building decision boundaries.
        
    Returns: 
        None
    '''

    files=[]

    if not directory:
        print('You must specify a file directory to save the image files and .gif file')
    
    h = .02  # step size in the mesh
    fig = plt.figure(figsize=(8, 8))
    i = 1

    # preprocess dataset, split into training and test part
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.cool
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = fig.add_subplot(1,1,1)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for num, clf in enumerate(classifiers):
        ax = fig.add_subplot(1,1,1)
        score = clf.score(X_test, y_test)
        score = round(score,2)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')

        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        ax.set_title('Model Accuracy: '+str(score))
        ax.text(xx.max() - .3, yy.min() + .3, ' ', size=15, horizontalalignment='right')
        i += 1

        #plt.tight_layout()
        #plt.show()
        file=directory+str(num)+'.png'
        fig.savefig(file)
        files.append(file)
        
    make_gif(files, directory)
    
    return