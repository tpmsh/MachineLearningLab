"""
Experiment 8:
Classification using Decision Trees
Problem Statement:
Predicting whether a person is COVID-19 infected or not
"""

feature_names = ['Fever', 'Cough', 'Breathing Issue']  
X = [ [0,0,0], [1,1,1], [1,1,0],
      [1,0,1], [1,1,1], [0,1,0],
      [1,0,1], [0,1,1], [1,1,0],
      [0,1,0],[0,1,1],[0,1,1],[1,1,0] ] 
# Labels for the training data 
Y = ['NO', 'YES', 'NO', 'YES', 
     'YES', 'NO', 'YES', 'YES',
     'YES', 'NO', 'YES', 'NO', 'NO'] 
# Finding Unique Labels 
labels = list(set(Y)) 
# Create a decision tree classier model 
from sklearn.tree import DecisionTreeClassifier as DTClf
clf = DTClf(max_depth =3).fit(X,Y)

# Now predict and print the class for the unknown patient(example) 
pred = clf.predict([[1,1,0]]) 
print(f"Person infected? {pred[0]}") 
# plotting tree for COVID-19 classication 
import matplotlib.pyplot as plt 
# plt the figure,  
plt.figure(figsize=(30,20)) 
from sklearn.tree import plot_tree as PT
PT(clf, class_names=labels,
   feature_names=feature_names,
   rounded = True,filled = True,fontsize=14) 
plt.title('Decsion Tree for Predicting the Covid'
          + ' Infection(The left node is True)') 
plt.show()
