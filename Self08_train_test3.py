import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from sklearn.model_selection import train_test_split

x_train,y_train,x_test,y_test=train_test_split(
    x,y,test_size=0.3,
    random_state=1234,
    shuffle=True )

print(x_train)
print(x_test)

model=Sequential()
model.add(Dense(1,input_dim=1))

model.compile(loss="mae",optimizer="adam")
model.fit(x_train,y_train,epochs= 100)

loss=model.evaluate(x_test,y_test)
print('loss:',loss)

result=model.predict([11])
print("[11]의값",result)
