import numpy as np 
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1 )

#지금시간 8시 7분
