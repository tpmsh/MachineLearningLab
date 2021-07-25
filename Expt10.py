from sklearn.datasets import load_wine
wine = load_wine()

# printing class distribution of wine dataset
import numpy as np
print(f'Classes: {np.unique(wine.target)}')
print(f'Class distribution of the dataset: {np.bincount(wine.target)}') 

# from sklearn.cross_validation import train_test_split(Hold out method)
from sklearn.model_selection import train_test_split as SPLIT
X_train, X_test, y_train, y_test = SPLIT(wine.data, wine.target, test_size=0.25,stratify=wine.target, random_state=123) 

# printing class distribution of test dataset
print(f'Classes: {np.unique(y_test)}')
print(f'Class distribution for test data: {np.bincount(y_test)}') 

# MLP is sensitive to feature scaling, hence performing scaling
# Options: MinmaxScaler and Standardscaler
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import StandardScaler as SS
X_train_stdsc = SS().fit_transform(X_train)
X_test_stdsc = SS().fit_transform(X_test)

# Setting of hyperparameters of the network
from sklearn.neural_network import MLPClassifier as MLP
mlp = MLP(hidden_layer_sizes=(10,),learning_rate_init=0.001,max_iter=5000)

# Calculating Training Time : more neurons, more time 
from time import time
start = time()
# Train the model using the scaled training sets
mlp.fit(X_train_stdsc, y_train)
end = time()
print(f'Training Time: {(end-start)*1000:.3f}ms')

# Predict the response for test dataset
y_pred = mlp.predict(X_test_stdsc) # scaled

# Import scikit-learn metrics module for evaluating model performance
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Model Accuracy, how often is the classifier correct?
print(f'Accuracy:{accuracy_score(y_test, y_pred)}')

# display the confusion matrix
print('Confusion Matrix is:\n',confusion_matrix(y_test, y_pred))
print('Classification Report:\n',classification_report(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix as PCM
PCM(mlp, X_test_stdsc, y_test)