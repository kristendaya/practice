import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1])

print(x)
print(y)

x_train=np.array([1,2,3,4,5,6,7])
y_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_test=np.array([8,9,10])

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense (6))
model.add(Dense (3))
model.add(Dense (1))

model.compile(loss="mae",optimizer="adam")
model.fit(x_train,y_train,epochs=100, batch_size=2)

loss=model.evaluate(x_test,y_test)
print("loss:",loss)
result=model.predict([11])
print(result)

#[[11.022964]]
