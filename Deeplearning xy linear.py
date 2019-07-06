#!/usr/bin/env python
# coding: utf-8

# In[386]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 200
import random
import math
from datetime import datetime
from numpy import *
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
random.seed()


# In[387]:


array_len = 100
range_width = 1000

first_input = np.empty(array_len)
second_input = np.empty(array_len)
for i in range(array_len):
  first_input[i] = random.randint(0, range_width)
  second_input[i] = random.randint(0, range_width)

x = np.vstack((first_input, second_input)).T
x1 = first_input*2 + second_input*3 + 4
x2 = first_input*5 - second_input*6 - 7
y = x1*8+ x2*9 + 10
print("Done.")


# In[398]:


#training_frac = 0.85
#train_max_index = math.floor(array_len * training_frac)
#X_train = x[:train_max_index,:]
#Y_train = y[:train_max_index]
#X_test = x[train_max_index:,:]
#Y_test = y[train_max_index:]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)
print("Done.")


# In[401]:


model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.01), loss='mse')
model.fit(X_train, Y_train, validation_split = 0.2, batch_size=10, epochs=10000)
# Note: sometimes this works with 5 nodes in the first layer, sometimes it doesn't


# In[402]:


Y_predict = model.predict(X_test)
model.evaluate(X_test, Y_test)


# In[403]:


plt.plot(X_test[:,0], Y_test, linestyle='none', marker='.', color='green', label='Test data')
plt.plot(X_test[:,0], Y_predict, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[404]:


plt.plot(X_test[:,1], Y_test, linestyle='none', marker='.', color='green', label='Test data')
plt.plot(X_test[:,1], Y_predict, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[405]:


for layer in model.layers:
    print(layer.get_weights())

#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][0]) + model.layers[0].get_weights()[1][0]
#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][1]) + model.layers[0].get_weights()[1][1]
#blob3 = blob1*model.layers[1].get_weights()[0][0] + blob2*model.layers[1].get_weights()[0][1] + model.layers[1].get_weights()[1]
#print(blob3)


# In[406]:


fig=p.figure()
ax = p3.Axes3D(fig)
ax.scatter(xs=X_test[:,0], ys=X_test[:,1], zs=Y_test, zdir='z', s=20, c=None, depthshade=True, color='blue')
ax.scatter(xs=X_test[:,0], ys=X_test[:,1], zs=Y_predict, zdir='z', s=20, c=None, depthshade=True, color='red')


# In[ ]:




