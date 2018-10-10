
# coding: utf-8

# In[94]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[95]:


data = pd.read_csv("years-lived-with-disability-vs-health-expenditure-per-capita.csv",delimiter=',')


# In[96]:


data = data[data.Year == 2011]


# In[97]:


# data.head()


# In[186]:


# plt.scatter(data.Health_expenditure_per_capita_PPP,data.Years_Lived_With_Disability)


# In[99]:


x_obs = np.array(data.Health_expenditure_per_capita_PPP)
y_obs = np.array(data.Years_Lived_With_Disability)


# In[154]:


def model(x,m,b):
    return m*x+b


# In[190]:


def dist(x,y,m,b):
    A = -1
    B = 1/m
    C = -b/m
    return abs(A*x+B*y+C)/((A**2+B**2)**0.5)

def loglikelihood(x_obs, y_obs, sigma_y_obs, m, b):
#     d = y_obs -  model(x_obs, m, b)
#     d = d/sigma_y_obs
#     d = -0.5 * np.sum(d**2)
    d = dist(x_obs,y_obs,m,b)
    sigma = 1
    d = -0.5 * np.sum(d**2)
    return d

def logprior(m, b):
    p = -np.inf
    if m < 1/4000 and m >0 and b >-10 and b<10:
        p = 0.0
    return p


# In[191]:


N = 300000
lista_m = [np.random.random()]
lista_b = [np.random.random()]
logposterior = [loglikelihood(x_obs, y_obs, 1, lista_m[0], lista_b[0]) + logprior(lista_m[0], lista_b[0])]

sigma_delta_m = 0.0001
sigma_delta_b = 0.01

for i in range(1,N):
    propuesta_m  = lista_m[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_m)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)

    logposterior_viejo = loglikelihood(x_obs, y_obs, 5, lista_m[i-1], lista_b[i-1]) + logprior(lista_m[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, 5, propuesta_m, propuesta_b) + logprior(propuesta_m, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_m.append(propuesta_m)
        lista_b.append(propuesta_b)
        logposterior.append(logposterior_nuevo)
    else:
        lista_m.append(lista_m[i-1])
        lista_b.append(lista_b[i-1])
        logposterior.append(logposterior_viejo)
lista_m = np.array(lista_m)
lista_b = np.array(lista_b)
logposterior = np.array(logposterior)


# In[192]:


yb,paramb = (np.histogram(lista_b,normed=True))
paramb=paramb[1:]

b = paramb[yb==max(yb)]

ym,paramm = (np.histogram(lista_m,normed=True))
paramm=paramm[1:]
m = paramm[ym==max(ym)]

plt.scatter(data.Health_expenditure_per_capita_PPP,data.Years_Lived_With_Disability)
xs = np.linspace(0,100,1000)
ys = model(xs,m,b)
plt.title("m = {:.2f} b = {:.2f}".format(np.mean(lista_m),np.mean(lista_b)))
plt.plot(xs,ys,c='orange')
plt.ylim(4,12)


# In[195]:


plt.hist(lista_m)


# In[196]:


def dist(x,y,m,b):
    A = -1
    B = 1/m
    C = -b/m
    return abs(A*x+B*y+C)/((A**2+B**2)**0.5)



def model2(x,a,b):
    return a*np.log(x) + b

def loglikelihood(x_obs, y_obs, sigma_y_obs, m, b):
#     d = y_obs -  model2(x_obs, m, b)
#     d = d/sigma_y_obs
#     d = -0.5 * np.sum(d**2)
    d = dist(x_obs,y_obs,m/np.log(x_obs),y_obs+m*x_obs/np.log(x_obs))
    sigma = 1
    d = -0.5 * np.sum(d**2)
    return d

def logprior(m, b):
    p = -np.inf
    if m < 2 and m >0.1 and b >0 and b<3:
        p = 0.0
    return p


# In[205]:


N = 200000
lista_m = [np.random.random()]
lista_b = [np.random.random()]
logposterior = [loglikelihood(x_obs, y_obs, 1, lista_m[0], lista_b[0]) + logprior(lista_m[0], lista_b[0])]

sigma_delta_m = 0.01
sigma_delta_b = 0.01

for i in range(1,N):
    propuesta_m  = lista_m[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_m)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    while(propuesta_m<0 or propuesta_b<0):
        propuesta_m  = lista_m[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_m)
        propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)

    logposterior_viejo = loglikelihood(x_obs, y_obs, 0.1, lista_m[i-1], lista_b[i-1]) + logprior(lista_m[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, 0.1, propuesta_m, propuesta_b) + logprior(propuesta_m, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_m.append(propuesta_m)
        lista_b.append(propuesta_b)
        logposterior.append(logposterior_nuevo)
    else:
        lista_m.append(lista_m[i-1])
        lista_b.append(lista_b[i-1])
        logposterior.append(logposterior_viejo)
lista_m = np.array(lista_m)
lista_b = np.array(lista_b)
logposterior = np.array(logposterior)


# In[206]:


yb,paramb = (np.histogram(lista_b,normed=True))
paramb=paramb[1:]

b = paramb[yb==max(yb)]

ym,paramm = (np.histogram(lista_m,normed=True))
paramm=paramm[1:]
m = paramm[ym==max(ym)]

print(m,b)

plt.scatter(data.Health_expenditure_per_capita_PPP,data.Years_Lived_With_Disability)
xs = np.linspace(0.02001,8000,1000)
ys = model2(xs,m,b)
plt.title("m = {:.2f} b = {:.2f}".format(m[0],b[0]))
plt.plot(xs,ys,c='orange')
# plt.ylim(4,13)


# In[210]:


plt.hist(lista_b,bins=40)

