
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

x = np.array(
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ])
    
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
  
  # PREDICT [[10, 1.4, 0]]
  
trans_x=x.transpose()

print(trans_x)
  model=Sequential()
  model.add(Dense(1, input_dim=3))
  model.add(Dense(3))
  model.add(Dense(6))
  model.add(Dense(3))
  model.add(Dense(1))

model.compilie(loss="mae",optimizer="adam")
model.fit(trans_x,epochs=300)

#evaluate
loss=model.evaluate(trans_x,y)
print("loss:",loss)

#result
result=model.predict([[10,1.4,0]])
print("result:", result)





  
  