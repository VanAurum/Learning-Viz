#Standard python imports
from sklearn.datasets import make_moons

#Local imports
from learning_viz.learning_viz import plot_decision_bounds
from learning_viz.make_models import generate_models


if __name__=='__main__':

    directory = '/Users/vanaurum/Dropbox/Learning Visualization/KNN02/'
    dataset = make_moons(n_samples = 500, noise=0.35, random_state=0)
    clfs, names = generate_models(dataset, model='KNN', iterations=40000)
    plot_decision_bounds(dataset, names, clfs, directory)