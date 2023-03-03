import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([range(10),range(21,31),range(201,211)])
print(x.shape)
new_x= x.T 

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
              
new_y = y.T

model=Sequential()
model.add(Dense (3,input_dim=3))
model.add(Dense (5))
model.add(Dense (6))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (1))

model.compile(lose="mae",optimizer="adam")
model.fit(new_x,new_y,epochs=100)

loss=model.evaluate(new_x,new_y)
print("loss:",loss)

result=model.predict([[3, 30, 210]])
print("result[[9, 30, 210]]:", result)
