#!/usr/bin/env python
# coding: utf-8

# linear regression

# In[31]:


import pandas as pd
df = pd.read_csv(r"Downloads/housing_data.csv")
df.head()


# In[32]:



X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
X


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg


# In[34]:


lin_reg.fit(X_train,Y_train)
lin_reg.predict(X_test)


# In[35]:


lin_reg.score(X_test,Y_test)


# In[36]:


lin_reg.coef_


# In[37]:


lin_reg.intercept_


# In[ ]:





# In[39]:


from sklearn.model_selection import cross_val_score
import numpy as np
mse = cross_val_score(lin_reg,X,Y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[ ]:




