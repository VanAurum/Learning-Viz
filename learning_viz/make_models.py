#Standard python imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from operator import itemgetter

#Local imports
from learning_viz.model_parameters import get_params

def generate_models(dataset, model='KNN', iterations=40000):
    '''Creates a list of trained models and their scores to be plotted.

    Parameters:
    -----------
    dataset : object 
        X, y training data from an sklearn toy dataset
    model : string (optional, default = 'KNN')
        The key of the learning algorithm to model.
    iterations : int (optional, default=40000)
        The number of models to build. The gif maker sorts this list by performance 
        and only takes 100 evenly spaced images to make the gif.  More iterations 
        will give you a better cross section of the learning process but takes longer. 

    Returns:
    --------
    clfs : list 
        A list of classifier objects whose decision bounds will be plotted.
    names : list
        A list of names to use as frame titles in each GIF sequence.        
    '''

    names = [] #  Store the images names to be used as the title in each GIF frame
    classifiers = [] #  The list of classifiers to plot probability bounds for.

    # Prepare toy dataset.
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.35, random_state=42)

    for i in range(1,iterations):
        print('iteration: '+str(i))
        params = get_params(model)
        clf = model_library(model, params)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)    
        classifiers.append([score, clf])
        names.append(str(i))
    
    # Sort by score, take 100 evenly spaced samples from this list to use as GIF frames.
    classifiers = sorted(classifiers, key=itemgetter(0))
    skip_every = int(len(classifiers)//100)
    clfs = classifiers[0:-1:skip_every]
    clfs = [x[1] for x in clfs]

    return clfs, names


def model_library(model, params):
    '''A library function to return appropriate model call.

    Parameters: 
    -----------
    model : string 
        The key for the model learning function to call
    params : dictionary 
        The parameters to pass to the learning function.

    Returns:
    --------
    A function call to the appropriate learning function        
    '''

    library = { 
        'KNN' : KNeighborsClassifier(**params),
        'RFC' : RandomForestClassifier(**params),
        'SVM' : SVC(**params),
    }

    return library[model]