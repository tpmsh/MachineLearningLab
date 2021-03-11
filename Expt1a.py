"""                 Expt. 1A
Linear regression using linear least squares fit method
"""
import matplotlib.pyplot as plt
from numpy import array, zeros
from statistics import mean
from sklearn.metrics import r2_score


# past data for training the regression model
# Example Predictor is CGPA and Target is Salary
cgpa = array(
    [6.1, 6.15, 6.30, 7.24,
     7.50, 7.50, 7.90, 8.0,
     8.9, 9.1, 9.5, 9.5, 9.52])
salary = array(
    [0.0, 2.50, 2.25, 6.00, 
     3.30, 3.75, 4.5, 3.30, 
     4.0, 3.5, 6.5, 10.5, 14.5])

# Finding slope and y_intercept 
# for the best fit line using formulae
average_cgpa, average_salary = mean(cgpa), mean(salary)
numerator, denominator = 0, 0

for i in range(len(cgpa)):
    numerator += (cgpa[i]-average_cgpa) * (salary[i]-average_salary)
    denominator += (cgpa[i]-average_cgpa) ** 2

slope = numerator / denominator
y_intercept = average_salary - slope * average_cgpa
print('Equation of the best fit line is: y = '
      + f'{slope:.2f} * x + {y_intercept:.2f}') 


# Computing R square value
predicted_salary = zeros((len(cgpa), 1))
for i in range(len(cgpa)):
    predicted_salary[i] = slope * cgpa[i] + y_intercept
score_r2 = round(r2_score(salary, predicted_salary), 2)
print('R2 Score:', score_r2)

#plot the training instances(points)
plt.scatter(cgpa, salary, color = 'blue', label = 'Training Data')

#Plotting the best fit line
plt.xlabel("CGPA") 
plt.ylabel("Salary")
plt.title(f"Linear Ordinary LS fit Model - R2 Score = {score_r2}")
plt.plot([5.5, 10.0], slope * array([5.5, 10.0]) + y_intercept, 'red', label = 'Best Fit Line')



#For the given CGPAs find the  predicted salary
predictor_cgpa = array([9.11, 5.25, 8.58, 7.26, 9.85])
response_salary = slope * predictor_cgpa + y_intercept
plt.plot(predictor_cgpa, response_salary, 'gs', label = 'Predicted Salary') 
for i in range(len(predictor_cgpa)):
    response_salary[i] = response_salary[i] if response_salary[i] > 0 else 0
    print(f'Predicted Salary for the Student {i+1} with CGPA '
          + f'{predictor_cgpa[i]} is Rs {response_salary[i]:.2f} Lac')

plt.legend()