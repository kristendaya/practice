import numpy as np 
x=np.array([1,2,3])
y=np.array([1,2,3])

#modeling
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model=Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(1))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#compile
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

#평가,예측
loss=model.evaluate(x,y)
print("loss:",loss)  #ㅠㅠㅠㅠ 바보 "" 는 출력하고 싶은것을 입력한다, 그래서 "loss" 하는 이유는 값을 쉽게보기위해서

result=model.predict([4])
print("[4]의 예측값",result)


