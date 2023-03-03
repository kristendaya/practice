#put data
import numpy as np
x=np.array([1,2,3])
y=np.array([1,2,3])

#model
import tensorflow as tf 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model=Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#compile
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)
