import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, MaxPooling1D,Conv1D, Dropout,GlobalAveragePooling1D
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import fasttext

#读取csv数据
train_data=pd.read_csv(r"C:\Users\WWG\Desktop\train_set.csv\train_set.csv",sep='\t')

train_x=list(train_data['text'])[0:150000]
train_y=list(train_data['label'])[0:150000]
train_y=tf.one_hot(train_y,max(train_y)+1)

print('训练集文本长度为： '+str(len(train_x)))


#切词操作
for i in range(len(train_x)):
    train_x[i]=list(map(int, train_x[i].split()))
#填充
zero_list=[]
for i in range(1000):
    zero_list.append(0)
for j in range(len(train_x)):
    train_x[j].extend(zero_list)
    train_x[j]=train_x[j][0:500]
train_x=tf.convert_to_tensor(train_x)

model=Sequential()
model.add(Embedding(input_dim=20000,output_dim=300,input_length=500))
model.add(LSTM(units=32))
model.add(Dense(units=14,activation='relu'))
model.compile(optimizer='rmsprop',loss="mean_squared_error",metrics = ['acc'])
history=model.fit(train_x,train_y,epochs=100,batch_size=1000)
model.save(filepath=r"C:\Users\WWG\Desktop\test_a.csv")

acc = history.history['acc']
loss = history.history['loss']
plt.subplot(1,2,1)
epochs = range(1, len(acc) + 1)
# plt.scatter(epochs, acc, s=11)
plt.plot(epochs,acc,color='green',linestyle='-',linewidth=1,label='Training Of Acc')
plt.text(x=epochs[-1],y=acc[-1],s=str(acc[-1]))
plt.title('Training Accuracy')
plt.subplot(1,2,2)
# plt.scatter(epochs, loss, s=11)
plt.plot(epochs,loss,color='green',linestyle='-',linewidth=1,label='Training Of loss')
plt.text(x=epochs[-1],y=loss[-1],s=str(loss[-1]))
plt.title('Training Loss')
plt.show()

