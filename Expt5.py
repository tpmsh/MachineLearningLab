from sklearn.datasets import load_boston
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

#Loading the Boston dataset
boston = load_boston()
x=boston.data[:,:]
y=boston.target
print(x.shape,y.shape)

tsize=0.30 #30% of total data is used for testing and 70% used for training

##splitting the dataset into training and testing sets,
#(parameter random state is fixed at some integer, to ensure the same train 
#and test sets across various runs)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=tsize,random_state=102)

###---Code using Scikit learn library for KNN regression
##Finding MSEs for different values of k 
maxk=int(math.sqrt(xtrain.shape[0])) #maximum value of k 
mse_val = [] #to store rmse values for different k
for K in range(1,maxk):
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(xtrain, ytrain)  #fit the model
    pred=model.predict(xtest) #make prediction on test set
    error = mean_squared_error(ytest,pred) #calculate rmse
    mse_val.append(error) #store rmse values
    print('MSE value for k= ' , K , 'is:', error)

##function to find elbow point
def find_elbow():
    inds=np.argsort(mse_val)
    for i in inds:
        diff1=mse_val[i-1]-mse_val[i]
        diff2=mse_val[i]-mse_val[i+1]
        if(diff1>0 and diff2<0):   
            break
    eb1=i+1
    return(eb1)

##plotting the elbow curve 
k=np.arange(1,maxk)
xl="k"
yl="MSE"
plt.xlabel(xl) 
plt.ylabel(yl)
plt.title("Elbow Curve")
plt.plot(k,mse_val)

##finding the k for the elbow point 
ke=find_elbow()
print("Best Value of k using elbow curve is ",ke)
plt.plot(ke,mse_val[ke-1],'rx')
plt.annotate("  elbow point", (ke,mse_val[ke-1]))
## Now with the best k(i.e. ke) predict the cost for the new house with given features
xnew=np.array([2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,
               6.4210e+00, 7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,
                   1.7800e+01, 3.9690e+02 ,9.1400e+00])
model = neighbors.KNeighborsRegressor(n_neighbors = ke)
model.fit(xtrain, ytrain)  #fit the model
xnew=xnew.reshape(1,-1)
hcost=model.predict(xnew)
print("Predicted price of the given house is {:.2f}".format(hcost[0]),"thousand dollars")