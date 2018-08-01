
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv('USA_Housing.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[12]:


df.columns


# In[32]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms',
     'Area Population']]


# In[28]:


y=df['Price']


# In[24]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[36]:


X_test.head()


# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


lm=LinearRegression()


# In[39]:


lm.fit(X_train,y_train)


# In[40]:


#predictions
predictions=lm.predict(X_test)


# In[41]:


predictions


# In[42]:


plt.scatter(y_test,predictions)


# In[44]:


#histogram
sns.distplot((y_test-predictions),bins=50)


# In[45]:


#regression evaluation metrics
from sklearn import metrics


# In[48]:


metrics.mean_absolute_error(y_test,predictions,multioutput='uniform_average')


# In[49]:


metrics.mean_squared_error(y_test,predictions)


# In[50]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))

