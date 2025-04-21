# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1: Import Required Libraries
2: Prepare the Dataset
3: Select Feature and Target Variables
4: Train the Decision Tree Regressor Model
5: Make Predictions
6: Visualize the Results
7: Interpret Results

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: TAMIL PAVALAN M
RegisterNumber:  212223110058
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
# Dataset
data = {
    'Position': [
        'Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager',
        'Country Manager', 'Region Manager', 'Partner', 'Senior Partner', 'C-level', 'CEO'
    ],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
# Creating DataFrame
df = pd.DataFrame(data)
# Features and target
X = df[['Level']]  # Independent variable
y = df['Salary']   # Dependent variable
# Creating and fitting the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)
# Predict salary for a given level (e.g., Level 6.5)
level_to_predict = 6.5
predicted_salary = model.predict([[level_to_predict]])
print(f"Predicted salary for level {level_to_predict} is: ${predicted_salary[0]:,.2f}")
# Visualization
X_grid = np.arange(min(X.Level), max(X.Level) + 0.01, 0.01).reshape(-1, 1)
y_pred_grid = model.predict(X_grid)

plt.scatter(X, y, color='red', label='Actual')
plt.plot(X_grid, y_pred_grid, color='blue', label='Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Employee Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

## Output:

![image](https://github.com/user-attachments/assets/e33b1880-323c-48b9-bda4-4077b56c736c)

![image](https://github.com/user-attachments/assets/3ff3acc3-57f5-4666-8e2f-b2193f6543b0)

![image](https://github.com/user-attachments/assets/99d0c814-db37-4fc2-9dea-ff03a9d23863)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
