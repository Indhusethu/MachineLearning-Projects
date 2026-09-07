#Given Historical data of a Ecommerce Shopping Website,we need to predict the annual spend of each customer, and also identify the features which influence the customer spending the most.

import numpy as numpy
import pandas as pd 
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Reading the dataset
dataset=pd.read_csv('Ecommerce_Customers.csv')
print(dataset.head())

#Identifying the number of rows and columns in the dataset
print(dataset.shape)
 
#Statistical summary of the Dataset
print(dataset.describe())

#Identifying the Datatype of each column
print(dataset.info())

#exploring each variable by a Univariate/Bivariate Analysis
#Time on Website Vs Yearly Amount Spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=dataset)
plt.show()
plt.savefig('Time_on_website.png')

#Average Session Length Vs Yearly Amount Spent
sns.jointplot(x='Avg. Session Length',y='Yearly Amount Spent',data=dataset)
plt.show()
plt.savefig('Avg.Session Length.png')

#Length of Membership Vs Yearly Amount Spent
sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=dataset)
plt.show()
plt.savefig('Length_of_Membership.png')
# Hence, from the plot Visualiztion, we can say that the Length of Membership has the Maximum Impact on the dependent Variable.

#Separating the Feature set and the Dependent Variable Set
X=dataset[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y=dataset['Yearly Amount Spent']

#Splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=45)

#Initializing the Model
lm=LinearRegression()

#Fitting the model
lm.fit(X_train,y_train)

#The Predicted value of the coeffcients
print("The Coefficient of the Pred.Value is:", lm.coef_)

#The Predicted value of beta0
print("The Intercept of the Predicted Value is:", lm.intercept_)

#Storing the Predictions on the test_data into an object called Predictions
predictions=lm.predict(X_test)

#Evaluation Metrics using R_square
r2=r2_score(y_test,predictions)
print("The Value of Rsquare is:",r2)