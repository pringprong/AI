#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


#training_frac = 0.85
#train_max_index = math.floor(array_len * training_frac)
#X_train = x[:train_max_index,:]
#Y_train = y[:train_max_index]
#X_test = x[train_max_index:,:]
#Y_test = y[train_max_index:]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)
print("Done.")


# In[ ]:


model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.01), loss='mse')
model.fit(X_train, Y_train, validation_split = 0.2, batch_size=10, epochs=10000)
# Note: sometimes this works with 5 nodes in the first layer, sometimes it doesn't


# In[ ]:


Y_predict = model.predict(X_test)
model.evaluate(X_test, Y_test)


# In[ ]:


plt.plot(X_test[:,0], Y_test, linestyle='none', marker='.', color='green', label='Test data')
plt.plot(X_test[:,0], Y_predict, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


plt.plot(X_test[:,1], Y_test, linestyle='none', marker='.', color='green', label='Test data')
plt.plot(X_test[:,1], Y_predict, linestyle='none', marker='x', color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


for layer in model.layers:
    print(layer.get_weights())

#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][0]) + model.layers[0].get_weights()[1][0]
#blob1 = np.dot(X_test[0,:], model.layers[0].get_weights()[0][1]) + model.layers[0].get_weights()[1][1]
#blob3 = blob1*model.layers[1].get_weights()[0][0] + blob2*model.layers[1].get_weights()[0][1] + model.layers[1].get_weights()[1]
#print(blob3)


# In[ ]:


fig=p.figure()
ax = p3.Axes3D(fig)
ax.scatter(xs=X_test[:,0], ys=X_test[:,1], zs=Y_test, zdir='z', s=20, c=None, depthshade=True, color='blue')
ax.scatter(xs=X_test[:,0], ys=X_test[:,1], zs=Y_predict, zdir='z', s=20, c=None, depthshade=True, color='red')
#ax.legend()

ax.set_xlabel('$x1$', fontsize=20)
ax.set_ylabel('$x2$', fontsize=20)
#ax.yaxis._axinfo['label']['space_factor'] = 3.0
# set z ticks and labels
#ax.set_zticks([-2, 0, 2])
# change fontsize
#for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
# disable auto rotation
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$y$', fontsize=30, rotation = 0)


# In[ ]:


histories = []
num_trials = 40
for h in range(num_trials):
    print(h)
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(2,)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'], loss='mse')
    history = model.fit(X_train, Y_train, validation_split = 0.2, batch_size=10, epochs=2000)
    histories.append(history)


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 200
for j in range(num_trials):
    history = histories[j]
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


at1000 = []
for j in range(num_trials):
    at1000.append(histories[j].history['loss'][1000])
at1000.sort()
print(at1000)


# In[ ]:


# Next: try to display the model fits using FacetGrid
import seaborn as sns
sns.set(style="ticks", color_codes=True)
g = sns.FacetGrid(histories[0], col="loss", col_wrap=5, height=1.5)


# In[ ]:




