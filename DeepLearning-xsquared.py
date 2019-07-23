#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.initializers
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 200
import random
import math
from datetime import datetime
random.seed()


# In[ ]:


array_len = 10000
x = np.random.sample(array_len)
x = x*10 - 5
y = np.empty(array_len)

# discontinuous function:
#for i in range(array_len):
#    if (x[i] < 0.5):
#        y[i] = 0
#    elif (x[i] < 1000):
#        y[i] = 1
#    else:
#         y[i] =0

# Cubic function:
y = 0.25*(x**3) + 0.75*(x**2) - 1.5*x - 2
print ("Done.")


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)
print("Done.")


# In[ ]:


model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error')
model.fit(X_train, Y_train, epochs=300, validation_split=0.2, batch_size=100)


# In[ ]:


Y_predict = model.predict(X_test)
model.evaluate(X_test, Y_test)


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 200
plt.plot(X_train, Y_train, linestyle='none', marker='.', color='blue', label='Training data')
plt.plot(X_test, Y_test, linestyle='none', marker='.', color='cyan', label='Test data')
plt.plot(X_test, Y_predict, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


print(model.get_weights())


# In[ ]:


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def normalized_sigmoid_fkt(a, b, x):
   '''
   Returns array of a horizontal mirrored normalized sigmoid function
   output between 0 and 1
   Function parameters a = center; b = width
   '''
   s= 1/(1+np.exp(b*(x-a)))
   return 1*(s-min(s))/(max(s)-min(s)) # normalize function to 0-1

def logistic(x, minimum, maximum, slope, ec50):
    return maximum + (minimum-maximum)/(1 + (x/ec50)**slope)


# In[ ]:


array_len = 10000
range_width = 1000
x2 = np.random.random_sample((array_len,))

ec501 = 0.75
slope1 = -20
minimum1 = 0
maximum1 = 1

ec502 = 0.25
slope2 = 10
minimum2 = 0
maximum2 = 1
y2 = np.empty(array_len)

for i in range(array_len):
    #y2[i] =logistic(x2[i], minimum, maximum, slope, ec50)
    y2[i] = logistic(x2[i], minimum1, maximum1, slope1, ec501) + logistic(x2[i], minimum2, maximum2, slope2, ec502) -1        
print ("Done.")


# In[ ]:


X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x2, y2, test_size=0.15)
print("Done.")


# In[ ]:


model2 = Sequential()
model2.add(Dense(2, activation='sigmoid', input_shape=(1,)))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')
model2.fit(X_train2, Y_train2, validation_split = 0.2, batch_size=100, epochs=300)
# Note: this does not converge 100% of the time. may have to run more than once


# In[ ]:


Y_predict2 = model2.predict(X_test2)
model2.evaluate(X_test2, Y_test2)


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 200
plt.plot(X_train2, Y_train2, linestyle='none', marker='.', color='blue', label='Training data')
plt.plot(X_test2, Y_test2, linestyle='none', marker='.', color='cyan', label='Test data')
plt.plot(X_test2, Y_predict2, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


for layer in model2.layers:
    print(layer.get_config())
    print(layer.get_weights())


# In[ ]:




