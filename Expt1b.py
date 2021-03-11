"""           Expt. 1B
Linear regression with Ordinary least squares method 
using Scikit-learn
"""
import matplotlib.pyplot as plt
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# past data for training the regression model
cgpa = array([6.1, 6.15, 6.30, 7.24, 7.50,
     7.50, 7.9, 8.0, 8.9, 9.1, 
     9.5, 9.5, 9.52]).reshape(-1,1)
salary = array([0.0, 2.50, 2.25, 6.00, 3.30, 
     3.75, 4.5, 3.3, 4.0, 3.5, 
     6.5, 10.5, 14.5]).reshape(-1,1)

#plot the training instances(points)
plt.scatter(cgpa, salary, color='blue', label = 'Training Data')

#Defining and fitting the model
ols = LinearRegression()
ols.fit(cgpa, salary)

#Visualizing the linear model and Plotting the best fit line
plt.xlabel("CGPA") 
plt.ylabel("Salary")
plt.title("Linear Ordinary LS fit Model - R2 Score "
          +f" = {ols.score(cgpa, salary)}")
plt.plot([5.5, 10.0], ols.predict(array([5.5, 10.0]).reshape(-1,1)), 'red', label = 'Best Fit Line')
print(f'MSE = {mse(salary, ols.predict(cgpa)):.2f} ' 
      + f'R2 value =  {ols.score(cgpa, salary):.2f}')

#For the given CGPAs find the  predicted salary
predictor_cgpa = array([9.11, 5.25, 8.58, 7.26, 9.85]).reshape(-1,1)
response_salary = ols.predict(predictor_cgpa)
plt.plot(predictor_cgpa, response_salary, 'gs', label = 'Predicted Salary')
for i in range(len(predictor_cgpa)):
    response_salary[i] = response_salary[i] if response_salary[i] > 0 else 0
    print(f'Predicted Salary for the Student {i+1} with CGPA '
      + f'{predictor_cgpa[i][0]} is Rs {response_salary[i][0]:.2f} Lac')
plt.legend()