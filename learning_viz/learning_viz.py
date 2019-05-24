"""
Learning Viz : Visualizing the ML learning process
@author: Kevin Vecmanis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles, make_classification, make_multilabel_classification
from sklearn.model_selection import (StratifiedKFold, RepeatedKFold, KFold, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
import getpass
import imageio
from operator import itemgetter
import random

warnings.filterwarnings("ignore")



def parameters(algorithm):
    
    params = { 
            'RFC' : {
                    'n_estimators': np.random.choice(np.arange(5,15),1)[0],
                    'min_samples_leaf': np.random.choice(np.arange(0.005,0.05,0.005),1)[0],
                    'min_samples_split' : np.random.choice(np.arange(0.005,0.1,0.01),1)[0],
                    'max_depth': np.random.choice(np.arange(2,8),1)[0],
                    'n_jobs': -1,
                    },
                    
            'SVC':  {
                    'C': np.random.choice(np.arange(0.01, 1.0, 0.001),1)[0],
                    'gamma': 'auto',
                    'kernel': random.choice(['poly','rbf']),
                    'degree': np.random.choice([2, 3, 4, 5, 6], 1)[0],                 
                    },   
                    
            'KNN':  {
                    'n_neighbors': random.choice(list(range(1,40))),
                    'weights': random.choice(['uniform','distance']),
                    'algorithm': random.choice(['ball_tree','kd_tree','brute','auto']),
                    },                      
            }
    
    return params[algorithm]    

    

def plot_decision_bounds(dataset, names, classifiers, file_class):
    
    '''
    This function takes in a list of classifier variants and names and plots the decision boundaries
    for each on three different datasets that different decision boundary solutions.
    
    Parameters: 
        names: list, list of names for labelling the subplots.
        classifiers: list, list of classifer variants for building decision boundaries.
        
    Returns: 
    
        None
    '''
    images=[]
    userid=getpass.getuser() 
    directory='/Users/'+userid+'/Dropbox/Learning Visualization/'+file_class

    h = .02  # step size in the mesh
    fig = plt.figure(figsize=(8, 8))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
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
        #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax = fig.add_subplot(1,1,1)
        if ds_cnt == 0:
            ax.set_title("Input data")
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
            #clf.fit(X_train, y_train)
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
            if ds_cnt == 0:
                ax.set_title('Model Accuracy: '+str(score))
            ax.text(xx.max() - .3, yy.min() + .3, ' ',
                    size=15, horizontalalignment='right')
            i += 1
    
            #plt.tight_layout()
            #plt.show()
            file=directory+str(num)+'.png'
            fig.savefig(file)
            
            images.append(imageio.imread(file))
    imageio.mimsave(directory+'KNN.gif', images, fps=5)
    
    return