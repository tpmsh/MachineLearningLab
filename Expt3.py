import numpy as np
#Loading the dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()

# Print the information contained within the dataset
print(f"\nKeys of iris_dataset: \n{data.keys()}")
print(data['DESCR'][:500] + "\n...")

#Print the feature names
print(f"\nFeature names: \n{data['feature_names']}")

#Print the classes
print(f"\nTarget names: {data['target_names']}")

#Printing first Few Rows
print(f"\nFirst five rows of data:\n{data['data'][:5]}")

#Print the class values few datapoints
print(f"\nTarget:\n{data['target'][:5]}")

#Print the dimensions of data
print(f"\nShape of data: {data['data'].shape}")

X_train, X_test, Y_train, Y_test = train_test_split(
        data.data, data.target, 
        stratify=data.target,test_size=0.25) 

# Hyper Parameters
metr='euclidean' 
#other options:{'euclidean','minkowski','manhattan'},etc.
K=1

#Create KNN Classifiers
knn = KNeighborsClassifier(n_neighbors=K,metric=metr);   

#Train the classifier model using the training set
knn.fit(X_train, Y_train)

#Predict the response for test data x_new(dimensions of the new iris flower to be classified)
x_new = np.array([[5, 2.9, 1, .2]])
y_pred = knn.predict(x_new)

print(f"\nTest Data: {x_new} Prediction: {y_pred}") 
print(f"Predicted target name: {data['target_names'][y_pred]}")

# MODEL EVALUATION: Predict the responses for test dataset and calculate model accuracy
# Model Accuracy: how often is the classifier correct?
Y_pred = knn.predict(X_test)
print(f"\nTest set predictions:\n {Y_pred}")
print(f"\nK={K}, Distance metric = {metr} \n%Accuracy on Test set = {np.mean(Y_pred == Y_test)*100:.2f}")
