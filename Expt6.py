'''Expt. 6:Naïve Bayes Classifier using Scikit learn library on Wine Dataset.
https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
Problem Statement: Wine classification  is a very famous multi-class classification problem. 
The Wine dataset is the result of a chemical analysis of wines grown in the same region in Italy 
but derived from three different cultivars. The Dataset comprises of 13 features 
(alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, 
nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline) 
and type of wine cultivar. This data has three type of wine Class_0, Class_1, and Class_2. 
Build a ML model to classify the type of wine.
'''

# Import scikit-learn dataset library
from sklearn.datasets import load_wine
wine = load_wine()

from sklearn.model_selection import train_test_split as SPLIT
X_train, X_test, y_train, y_test = SPLIT(wine.data, wine.target, test_size=0.3,random_state=1) 
# 70% training and 30% test

def explore():
    # print the names of the 13 features
    print ("Features: ", wine.feature_names)
    
    # print the label type of wine(class_0, class_1, class_2)
    print ("Labels: ", wine.target_names)
    
    # print data(feature)shape
    wine.data.shape
    # print the wine data features (top 5 records)
    print (wine.data[0:5])
    print (wine.target)
    #Count number of observation in each class
    for i in set(wine.target):
        print('Class', i, ' -> ', list(wine.target).count(i))
        
explore()

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB as GNB

# Train the model using the training sets
# and Predict the response for test dataset
y_pred = GNB().fit(X_train, y_train).predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print(f"Accuracy: {100*metrics.accuracy_score(y_test, y_pred):.3f}%")

import matplotlib.pyplot as plt
from dabl import plot
from dabl.utils import data_df_from_bunch

plot(data_df_from_bunch(wine), 'target')
plt.show()