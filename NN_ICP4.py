#!/usr/bin/env python
# coding: utf-8

# 1)Data Manipulation
# a,b)Read the provided CSV file ‘data.csv’
# c)Show the basic statistical description about the data.

# In[2]:


import pandas as pd
df = pd.read_csv('data.csv')
df.describe() # Description statistical of the data


# d.) Check if the data has null values

# In[3]:


df.isnull().sum() # Checks if there are any null values


# 1). Replace the null values with the mean

# In[4]:


df['Calories'].fillna(df['Calories'].mean(),inplace=True) # Replace the null values with mean
df['Calories'].isnull().sum() # Checks if null still exists


# e.) Select at least two columns and aggregate the data using: min, max, count, mean

# In[5]:


df.groupby(['Duration','Pulse']).agg({'Calories':['min','max','count','mean'],'Maxpulse':['min','max','count','mean']})  # Aggregation of duration,pulse using calories and Maxpulse


# f.) Filter the dataframe to select the rows with calories values between 500 and 1000.

# In[6]:


df[(df['Calories'].between(500,1000))]  # Calories between 500 and 1000 data


# g.) Filter the dataframe to select the rows with calories values > 500 and pulse < 100.

# In[6]:


df[(df['Calories'] > 500) & (df['Pulse'] <= 100)] # Calories >500 and pulse<100 data


# h.) Create a new “df_modified” dataframe that contains all the columns from df except for
# “Maxpulse”

# In[7]:


df_modified=df.loc[:,df.columns!='Maxpulse']
df_modified  #  Df without maxpulse


# i.) Delete the “Maxpulse” column from the main df dataframe
# 

# In[8]:


df.drop('Maxpulse',axis=1) # Delete Maxpulse in main df


# j.) Convert the datatype of Calories column to int datatype.

# In[9]:


df['Calories']=df['Calories'].astype(int)#converting the data type to int
type(df['Calories'][0])


# k.) Using pandas create a scatter plot for the two columns (Duration and Calories).

# In[10]:


df.plot.scatter(x='Duration',y='Calories') #scatter plot


# 2. Linear Regression
# a) Import the given “Salary_Data.csv”

# In[11]:


ldf=pd.read_csv('Salary_Data.csv')
ldf.describe()   # Salary data description


# 
# b) Split the data in train_test partitions, such that 1/3 of the data is reserved as tes stubsets

# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(ldf.iloc[:, :-1].values,ldf.iloc[:,1].values,test_size =0.2)
x_train    # Checking train data


# c) Train and predict the model

# In[13]:


from sklearn.linear_model import LinearRegression
m=LinearRegression()#linearregression
m.fit(x_train, y_train)  # Fitting the data for the linear regression



# In[14]:


y_pred=m.predict(x_test)  # Predicting the data for testing 


# d) Calculate the mean_squared error

# In[15]:


import math
from sklearn.metrics import mean_squared_error as ms
ms(y_pred,y_test)#mean square error


# In[ ]:


e) Visualize both train and test data using scatter plot.


# In[16]:


import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)


# In[17]:


plt.scatter(x_test,y_test)


# In[ ]:




