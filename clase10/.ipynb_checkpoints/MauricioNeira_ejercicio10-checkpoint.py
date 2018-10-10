
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


data = pd.read_csv("years-lived-with-disability-vs-health-expenditure-per-capita.csv",delimiter=',')


# In[8]:


data = data[data.Year == 2011]
data = data.dropna()


# In[9]:


x_obs = np.array(data.Health_expenditure_per_capita_PPP)
y_obs = np.array(data.Years_Lived_With_Disability)






# ## Para M1

# In[26]:


m_min = 0
m_max = 1E-2
b_min = 0
b_max = 20

N=1000

ms = np.random.uniform(m_min,m_max,N)
bs = np.random.uniform(b_min,b_max,N)
def model(x,m,b):
    return m*x+b
def likelihood(x_obs, y_obs, m, b):
    ymodel = model(x_obs,m,b)
    gamm = ymodel*np.exp(-ymodel/y_obs)/(y_obs**2)
    return np.sum(gamm)

M1 = 0
for n in range(N):
    M1 += likelihood(x_obs,y_obs,ms[n],bs[n])
M1 /= N
M1


# ## Para M2

# In[33]:


def model2(x,a,b):
    return a*np.log(x) + b
def likelihood(x_obs, y_obs, m, b):
    ymodel = model2(x_obs,m,b)
    gamm = ymodel*np.exp(-ymodel/y_obs)/(y_obs**2)
    return np.sum(gamm)


m_min = 0.1
m_max = 2
b_min = -4
b_max = 12

N=1000

ms = np.random.uniform(m_min,m_max,N)
bs = np.random.uniform(b_min,b_max,N)


M2 = 0
for n in range(N):
    M2 += likelihood(x_obs,y_obs,ms[n],bs[n])
M2 /= N
M2


# ## Para M3

# In[34]:



def model3(x,a,b,c):
    return a*np.log(x-b) + c

def likelihood(x_obs, y_obs, m, b,c):
    ymodel = model3(x_obs,m,b,c)
    gamm = ymodel*np.exp(-ymodel/y_obs)/(y_obs**2)
    return np.sum((gamm))
    
m_min = 0.1
m_max = 2
b_min = -4
b_max = 12
c_min = -10
c_max = 10


N=1000

ms = np.random.uniform(m_min,m_max,N)
bs = np.random.uniform(b_min,b_max,N)
cs = np.random.uniform(c_min,c_max,N)
M3 = 0
for n in range(N):
    M3 += likelihood(x_obs,y_obs,ms[n],bs[n],cs[n])
M3 /= N
M3

    


# In[35]:


F12 = M1/M2
F13 = M1/M3
F23 = M2/M3
print(F12,F13,F23)

