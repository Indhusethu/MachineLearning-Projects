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
import statsmodels.formula.api  as sm



insurance=pd.read_csv('insurance.csv')
print(insurance.head())

#Here Expenses is our Dependent Variable and [age,sex,bmi,children,smoker and region] are our Independent Variable
insurance.info()

print(insurance.shape)

#Finding out missing values
print(insurance.isnull().sum())#There are no missing values in the dataset

#Checking the correlation
print(insurance.corr(numeric_only=True))

#To get the unique values of the Categorical Variables
obj=insurance.children.unique()
print("The Unique Values are:",obj)


#Uni-Variate Analysis
#Frequency Distribution Plot
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
insurance.plot(kind='hist',y='age',bins=70,color='b',ax=axes[0][0])
insurance.plot(kind='hist',y='bmi',bins=100,color='r',ax=axes[0][1])
insurance.plot(kind='hist',y='children',bins=4,color='g',ax=axes[1][0])
insurance.plot(kind='hist',y='expenses',bins=100,color='pink',ax=axes[1][1])
plt.show()
#plt.savefig('Histo_plot.png')

#Bi-Variate Analysis
#Scatterplot
sns.scatterplot(x='bmi',y='expenses',data=insurance,hue='smoker')
plt.show()
# plt.savefig('Scatter_plot.png')

#Pairplot Combines both Uni-variate and Bi-Variate  Analysis
sns.pairplot(data=insurance,hue='smoker')
plt.title('EDA with respect to Smoking Preferences')
plt.show()
# plt.savefig('Pair_plot.png')

#Countplot
sns.countplot(x='smoker',data=insurance)
plt.show()
# plt.savefig('Count_plot.png')

#Dummy Variable Creation
insurance=pd.get_dummies(data=insurance,columns=['sex','smoker','region'],drop_first=True)
print(insurance.head())

#Check the no.of.rows and Columns after creating dummy variables
print("The Shape of the Insurance dataset is :", insurance.shape)

#Check the datatype Info
print(insurance.info())

#Segragating all independent variables as X and dependent variables as y
X=insurance.drop('expenses',axis=1) #Independent Variable/Feature Set
y=insurance['expenses']

#Splitting the Feature  and Dependent variable set into training and Testing dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Concatenating the training data to create a single dataset
df=pd.concat([X_train,y_train],axis=1)
print(df.head())



#So,RSS and ESS are the erroe components as it captures the deviation of the actual values of y from the predicted values of y
#Ideally we want the differenec to be less as possible valiues(predicted=actuals),So our objective will always be
#to minimize RSS and ESS which are the squared deviations.Ordinary Least(min) square(Square of Deviations)

#Modeling our data
mlrstat=sm.ols(formula="""expenses ~age+bmi+children+sex_male+smoker_yes+region_northwest+region_southeast+region_southwest""",data=df).fit()
print(mlrstat.summary())

#p<0.05, we reject the null hypothesis.
#Model 2 
mlrstat=sm.ols(formula="""expenses ~age+bmi+children+smoker_yes+region_southeast+region_southwest""",data=df).fit()
print(mlrstat.summary())

#Model 3
mlrstat=sm.ols(formula="""expenses ~age+bmi+children+sex_male+smoker_yes""",data=df).fit()
print(mlrstat.summary())

#Multi-Collinearity Issue
#Impacts Model Performance
#Hard to eliminate Feature which has high P value
#Over estimating the importance of the related IDV
#The bias gets doubled if there is positive correation  and nullified if there is negative correlation
#We can judge the impact a single variable(among correlated variables) on the dependent variable
#Overfitting Problem

#Multicollinearity occurs when two or more independent variables have a high correlation with one another in a regression 
#model, which makes it difficult to determine the individual effect of each independent variable on the dependent variable.

### We use the VIF(Variance Inflation Factor) test to detect MC
probval=pd.DataFrame(round(mlrstat.pvalues,2))
from statsmodels.stats.outliers_influence import variance_inflation_factor
features=mlrstat.model.exog
vif=[variance_inflation_factor(features,i) for i in range(features.shape[1])]
print(vif)
probval['vif']=vif
print(probval)

#We will remove the VIF value greater than 2 , value gretaer than 2 possess mutlicollinearity issues



