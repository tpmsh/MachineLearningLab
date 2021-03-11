"""               Expt. 2 
Implementing linear classifier using non-parametric training approach
"""

import matplotlib.pyplot as plt
from numpy import array, dot
from statistics import mean
from numpy.linalg import norm

# First value in each vector is income(lacs) 
# Second is saving amount(lacs) per year

low_risk_customers = array([
    [7,0.7],[5.8,0.62],[5.5,0.5],[4.8,0.6],
        [7.5,0.6],[3.9,0.5],[3.5,0.6],
        [5.3,0.7],[9.2,0.6],[4.5,.7]])

high_risk_customers = array([
    [3,.05],[6.5,0.2],[4.5,0.12],[9.5,0.25],
        [4.8,0.03],[6,0.1],[4,0.15],
        [6.2,0.02],[5.4,.21],[7,0.16]])    

                                                              
#plot the low_risk training instances(points)
plt.scatter(low_risk_customers[:,0],low_risk_customers[:,1],color = 'blue', label = 'Low Risk Customers')

#plot the high_risk training instances(points)
plt.scatter(high_risk_customers[:,0],high_risk_customers[:,1],color = 'red', label = 'High Risk Customers')

#Computing cluster centers for both the classes

low_risk_avg_income = mean(low_risk_customers[:,0]) 
low_risk_avg_salary = mean(low_risk_customers[:,1])  
high_risk_avg_income = mean(high_risk_customers[:,0]) 
high_risk_avg_salary = mean(high_risk_customers[:,1])  

low_risk_cluster_center = array([low_risk_avg_income,low_risk_avg_salary])
high_risk_cluster_center = array([high_risk_avg_income,high_risk_avg_salary])

#plotting cluster center in green
plt.plot(low_risk_cluster_center[0],low_risk_cluster_center[1],'bs', label = 'Low Risk Cluster Center')
plt.plot(high_risk_cluster_center[0],high_risk_cluster_center[1],'rs', label = 'High Risk Cluster Center')

#Plotting line joining low_risk_cluster_center and high_risk_cluster_center(centroids)
Cx = [low_risk_cluster_center[0],high_risk_cluster_center[0]]
Cy = [low_risk_cluster_center[1],high_risk_cluster_center[1]]
plt.plot(Cx,Cy)

# #Finding g(x) of the decison line
W = low_risk_cluster_center - high_risk_cluster_center
norm1 = norm(low_risk_cluster_center)
norm2 = norm(high_risk_cluster_center)
b = 0.5 * ((norm2 ** 2) - (norm1 ** 2))

plt.xlabel("Income") 
plt.ylabel("Saving")
plt.title("Linear Classifcation Model")

# Applying the model on new data to predict class label
X = [7, 0.4]
g = dot(W,X)+b
if(g>0):
    print("Low risk customer ")
    plt.plot(X[0],X[1],'bx')
else:
    print("High risk customer ")
    plt.plot(X[0],X[1],'rx')

# Plotting Perpendicular Bisector
center = array([sum(Cx), sum(Cy)])/2
plt.plot(center[0], center[1], 'go')
slope = (Cy[0] - Cy[1]) / (Cx[0] - Cx[1])
slope = -1 / slope
intercept = center[1] - slope * center[0]
Px = array([3.0, 9.0])
Py = slope * Px + intercept
plt.plot(Px, Py, '--', label = 'g(x)')
#plt.legend(loc = 'center right')