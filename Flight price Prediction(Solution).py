#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv('flight_booking.csv')
df = df.drop(columns=['Unnamed: 0'])
df.head()


# In[6]:


df.shape
df.info()
df.describe()


# In[7]:


df.isnull().sum()


# In[13]:


sns.lineplot(x=df['airline'],y=df['price'],color='Orange')
plt.title('Airline VS price')
plt.xlabel('Airline')
plt.ylabel('Price')
plt.show()


# In[14]:


sns.lineplot(x=df['days_left'],y=df['price'],color='Blue')
plt.title('Days left vs Price')
plt.xlabel('Days left')
plt.ylabel('Price')
plt.show()


# In[18]:


sns.barplot(x=df['airline'],y=df['price'])
plt.show()


# In[21]:


sns.barplot(x=df['class'],y=df['price'],data=df,hue='airline')
plt.title('Price as per the Class of Flight')
plt.xlabel('Class')
plt.ylabel('Price')
plt.show()


# In[29]:


sns.lineplot(x=df['days_left'],y=df['price'],data=df,hue='source_city')
plt.show()


# In[30]:


sns.lineplot(x=df['days_left'],y=df['price'],data=df,hue='destination_city')
plt.show()


# In[31]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['airline']=le.fit_transform(df["airline"])
df['source_city']=le.fit_transform(df["source_city"])
df['departure_time']=le.fit_transform(df["departure_time"])
df['stops']=le.fit_transform(df["stops"])
df['arrival_time']=le.fit_transform(df["arrival_time"])
df['destination_city']=le.fit_transform(df["destination_city"])
df['class']=le.fit_transform(df["class"])
df.info()


# In[37]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming df is your DataFrame
col_list = []
for col in df.columns:
    if (df[col].dtype != 'object') and (col != 'Price'):
        col_list.append(col)

x = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

print(vif_data)


# In[40]:


df=df.drop(columns=['stops'])
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming df is your DataFrame
col_list = []
for col in df.columns:
    if (df[col].dtype != 'object') and (col != 'Price'):
        col_list.append(col)

x = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

print(vif_data)


# # Linear Regression

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Assuming df is your DataFrame
x = df.drop(columns=['price'])
y = df['price']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardizing the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Creating and training the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predicting the test set results
y_pred = lr.predict(x_test)

# Creating a DataFrame to compare actual and predicted values
difference = pd.DataFrame(np.c_[y_test, y_pred], columns=["Actual_value", "Predicted_value"])

difference


# In[1]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn import metrices
mean_abs_error=metrices.mean_absolute_error(y_test,y_pred)
mean_abs_error
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)
mean_sq_error=metrices.mean_squred_error(y_test,y_pred)
mean_sq_error
root_mean_sq_error=np.sqrt(metrices.mean_squred_error(y_test,y_pred))
root_mean_sq_error


# In[2]:


sns.distplot(y_test,label="Actual")
sns.displot(y_pred,label="Predicted")
plt.legend()


# In[ ]:




