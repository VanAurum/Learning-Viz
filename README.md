# Learning Viz
A library for capturing the machine learning training process as an animated gif.  This library is for learning purposes - I built it to make the process of model 'tuning' more intuitive for those just starting to learn about machine learning and data science

## Examples

### KNN Classifier learning decision boundary between red and blue class
![](gif_samples/KNN.gif)


### Random Forest Classifier learning decision boundary between red and blue class
![](gif_samples/RFC.gif)


### Support Vector Machine iterating through ranges of C parameter
![](gif_samples/SVM.gif)


## Usage

An example is provided for those that want to use the library.  (Keep in mind that you should use toy datasets for these visualizations)

```Python
#Standard python imports
from sklearn.datasets import make_moons

#Local imports
from learning_viz.learning_viz import plot_decision_bounds
from learning_viz.make_models import generate_models

directory = '<Your file path here>'
dataset = make_moons(n_samples = 500, noise=0.35, random_state=0)
clfs, names = generate_models(dataset, model='KNN', iterations=40000)
plot_decision_bounds(dataset, names, clfs, directory)
```