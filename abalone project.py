#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv


# In[57]:


import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.model_selection import train_test_split


# In[58]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv')
df


# In[59]:


df.head(8)


# In[60]:


df.tail(10)


# In[61]:


le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])


# In[62]:


df


# In[63]:


df.describe()


# In[64]:


df.info()


# In[65]:


df.isnull()


# In[66]:


df.isnull().sum()


# In[67]:


sns.heatmap(df.isnull())


# In[68]:


df.columns


# # checking skew

# In[69]:


df.skew()


# In[70]:


df['Height'].plot.hist()


# In[42]:


df['Whole weight'].plot.hist()


# In[43]:


df['Rings'].plot.hist()


# In[71]:


df.hist(grid=False,
       figsize=(10, 6),
       bins=30)


# In[73]:


df.shape


# In[74]:


for i in df.columns:
    plt.figure()
    sns.distplot(df[i])


# In[75]:


df.corr()


# In[76]:


sns.heatmap(df.corr())


# In[77]:


df.drop('Sex',axis=1,inplace=True)


# In[78]:


df


# In[79]:


df.corr()


# In[80]:


sns.heatmap(df.corr())


# In[84]:


df.plot(kind='box',subplots=True,layout=(3,9))


# In[86]:


df['Length'].plot.box()


# In[88]:


df.columns


# In[89]:


df['Diameter'].plot.box()


# In[90]:


df['Height'].plot.box()


# In[91]:


df['Whole weight'].plot.box()


# In[92]:


df['Shucked weight'].plot.box()


# In[93]:


df['Viscera weight'].plot.box()


# In[94]:


df['Shell weight'].plot.box()


# In[95]:


df['Rings'].plot.box()


# In[96]:


from scipy.stats import zscore
z=np.abs(zscore(df))
df


# In[97]:


threshold=3
print(np.where(z>3))


# In[98]:


df_new=df[(z<3).all(axis=1)]


# In[99]:


df_new


# In[102]:


x=df_new.iloc[:,0:-1]
x


# In[103]:


x.shape


# In[109]:


y=df_new.iloc[:,-1]
y


# In[110]:


y.shape


# In[151]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.40,random_state=45)


# In[152]:


x_train.shape


# In[153]:


y_train.shape


# In[154]:


x_test.shape


# In[155]:


y_test.shape


# In[156]:


lr=LinearRegression()


# In[157]:


lr.fit(x_train,y_train)


# In[158]:


lr.coef_


# In[159]:


lr.intercept_


# In[160]:


lr.score(x_train,y_train)


# In[161]:


pred=lr.predict(x_test)
print("predicted quality:",pred)
print("actual quality:",y_test)


# In[162]:


print('error:')
print('mean absolute error:',mean_absolute_error(y_test,pred))
print('mean squared error:',mean_squared_error(y_test,pred))
print('root mean squarred error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[ ]:




