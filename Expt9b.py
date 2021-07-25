from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Creating Linear SVC Model and Setting hyperparameters
kernel, C, gamma = "linear", 0.55, 0.001    
# gives mean acuracy 0.99 and std deviation 0.01

from sklearn.svm import SVC
model = SVC(kernel=kernel, C=C, gamma=gamma)
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Cross-Validation procedure
k = 5
cv = StratifiedKFold(n_splits=k ,shuffle=True)
# calling cross validation function
# evaluating score on each train-test fold
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
print(scores)
print(f"Mean accuracy with {kernel} kernel is {scores.mean()*100:.2f}%"
      + f" with a standard deviation of {scores.std():.2f}")