def explore():
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

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as SPLIT
data = load_iris()
#explore()
# Split dataset into training set and test set using stratification
X_train, X_test, Y_train, Y_test = SPLIT(data.data, data.target,
                                    stratify=data.target,test_size=0.2) 
# 20% data for testing
### The code for building the model and getting the accuracy ###
from sklearn.svm import SVC

# Create the SVC model with hyperparameters
kernel, C, gamma = "linear", 0.55, 0.0005 

model = SVC(kernel=kernel, C=C, gamma=gamma)

#Train the classifier model using the training set
clf = model.fit(X_train,Y_train)

# MODEL EVALUATION: Predict the responses for test dataset and calculate model accuracy
# Model Accuracy: how often is the classifier correct?
Y_pred = clf.predict(X_test)
print(f"Test set predictions: {Y_pred}")
print(f"Kernel={kernel},%Accuracy on Test set = {np.mean(Y_pred == Y_test)*100:.2f}")
# display the confusion matrix
from sklearn.metrics import confusion_matrix as CMat
print('Confusion Matrix is:\n',CMat(Y_test, Y_pred))

# Predict the response for test data x_new
# (dimensions of the new iris flower to be classified)
x_new = np.array([[5, 2.9, 1, .2]])
y_pred = clf.predict(x_new)

print(f"New Test Data: {x_new}\nPrediction:Class {y_pred}") 
print(f"Predicted target name: {data['target_names'][y_pred][0]}")
