from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as SPLIT
from numpy import sqrt, sum, argsort, arange, array, average
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

boston = load_boston() # Loading the Boston dataset
x = boston.data[:,:]
y = boston.target
print(x.shape, y.shape)
# fixed random state ensures same train and test sets across various runs
# Size : 30% test && 70% train 
xtrain, xtest, ytrain, ytest = SPLIT(x, y, test_size=0.3, random_state=102)
    
##function to find Euclidean distance
def edist(v1,v2):
    return sqrt(sum((v1-v2)**2))
 
def explore():
    ## Exploring the dataset characteristics and having glimpse of data
    # printing the sizes of training and testing data sets
    print(xtrain.shape,ytrain.shape)
    print(xtest.shape,ytest.shape)
    # Print the information contained within the dataset
    print("\nKeys of iris_dataset: \n{}".format(boston.keys()))
    print(boston['DESCR'][:500] + "\n...")
    #Print the feature names
    print("\nFeature names: \n{}".format(boston['feature_names']))
    #Printing the  Few Rows
    print("\nFirst five rows of data:\n{}".format(boston['data'][:5]))
    #Print the class values few datapoints
    print("\nTarget:\n{}".format(boston['target'][:5]))
    #Print the dimensions of data
    print("\nShape of data: {}".format(boston['data'].shape))

##Function to find mean squared error for the entire test dataset
def knn_mse(k):
    preds = array([knn_reg(xtest[i] , k) for i in range(xtest.shape[0])])
    return MSE(ytest , preds)

##function to predict values using knn for given test data tx
def knn_reg(tx , k):
    #Find distances between new data and all the training data
    distances = array([edist(xtrain[i], tx) for i in range(xtrain.shape[0])])
    
    #sort the distances in ascending order
    inds = argsort(distances)
    distances = distances[inds]
    tr_y_sorted = ytrain[inds] #sorted values of target variable
    
    #predicted value is the average of first k values of target vector
    return average(tr_y_sorted[:k])
    
## function to find elbow plot
def find_elbow():
    inds=argsort(mse_val)
    for i in inds:
        diff1=mse_val[i-1]-mse_val[i]
        diff2=mse_val[i]-mse_val[i+1]
        if(diff1>0 and diff2<0):   
            break
    eb1=i+1
    return(eb1)

if __name__ == '__main__':
    explore()
    
    ##Finding MSEs for different values of k 
    maxk=int(sqrt(xtrain.shape[0])) #maximum value of k 
    mse_val = [] #to store rmse values for different k
    for k in range(1,maxk):
        error= knn_mse(k)
        mse_val.append(error) #store rmse values
        print(f'MSE value for k = {k} is: {error:.3f}')
        
    ##plotting the elbow curve 
    k=arange(1,maxk)
    plt.xlabel("k") 
    plt.ylabel("MSE")
    plt.title("Elbow Curve")
    plt.plot(k,mse_val, 'cyan')
    ##finding the k for the elbow point 
    ke=find_elbow()
    print(f"Best Value of k using elbow curve is {ke}")
    plt.plot(ke,mse_val[ke-1],'rx', label = 'elbow point')
    plt.legend()
    
    # Now model is ready to predict the cost for new house with 
    # given features in xnew vector and ke as k
    xnew=array([2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00,
                   4.6900e-01, 6.4210e+00, 7.8900e+01, 4.9671e+00, 
                   2.0000e+00, 2.4200e+02, 1.7800e+01, 3.9690e+02 ,9.1400e+00])
    hcost = knn_reg(xnew, ke)
    print(f"Predicted price of the given house is {hcost:.2f} thousand dollars")