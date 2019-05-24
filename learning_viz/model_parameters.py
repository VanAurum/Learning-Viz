
#Standard Python library imports
import numpy as np
import random

def parameters(algorithm):
    '''A library function to return a random parameter space given a key ('algorithm')
    '''
    
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