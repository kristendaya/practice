#[[10,1.4]]의 예측값
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

x=np.array([[1,1],
           [2,1],
           [3,1],
           [4,1],
           [5,1],
           [6,1],
           [7,1.3],
           [8,1.5],
           [9,1.6],
           [10,1.4]])
y=np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape)
print(y.shape)

#model
model=Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#compile,fit
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=30,batch_size=3)

#evaluation
loss=model.evaluate(x,y)
print=("loss:",loss)

#result
result=model.predict([[10,1.4]])
print=("예측값:",result)
