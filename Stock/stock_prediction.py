import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal


import pandas as pd
data = pd.read_csv('N225_2.csv')

df1 = pd.DataFrame(index=[])
df2 = pd.DataFrame(index=[])

df1['Close_LN_diff_100'] = data['Close_LN_diff_100']
df2['r>=1%'] = data['r>=1%']
df2['r>=0%'] = data['r>=0%']
df2['r>=-1%'] = data['r>=-1%']
df2['-1%>r'] = data['-1%>r']

# df1.head()
# df2.head()

# csv保存
df1.to_csv("x-data.csv", index = False)
df2.to_csv("y-data.csv", index = False)

df1_ = csv.reader(open('x-data.csv', 'r'))
data1 = [ v for v in df1_]
mat = np.array(data1)
mat2 = mat[1:]
x_data = mat2[:, :].astype(np.float)

df2_ = csv.reader(open('y-data.csv', 'r'))
data2 = [ v for v in df2_]
mat3 = np.array(data2)
mat4 = mat3[1:]
t_data = mat4[:, :].astype(np.float)



maxlen = 15              # 入力系列数
n_in = x_data.shape[1]   # 学習データ（＝入力）の列数
n_out = t_data.shape[1]  # ラベルデータ（=出力）の列数
len_seq = x_data.shape[0] - maxlen + 1
data = []
target = []
for i in range(0, len_seq):
  data.append(x_data[i:i+maxlen, :])
  target.append(t_data[i+maxlen-1, :])

x = np.array(data).reshape(len(data), maxlen, n_in)
t = np.array(target).reshape(len(data), n_out)

print(x.shape, t.shape)

n_train = int(len(data)*0.9)              # 訓練データ長
x_train,x_test = np.vsplit(x, [n_train])  # 学習データを訓練用とテスト用に分割
t_train,t_test = np.vsplit(t, [n_train])  # ラベルデータを訓練用とテスト用に分割

print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)


class Prediction :
  def __init__(self, maxlen, n_hidden, n_in, n_out):
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.n_in = n_in
    self.n_out = n_out

  def create_model(self):
    model = Sequential()
    model.add(LSTM(self.n_hidden, batch_input_shape = (None, self.maxlen, self.n_in),
             kernel_initializer = glorot_uniform(seed=2021), 
             recurrent_initializer = orthogonal(gain=1.0, seed=2021), 
             dropout = 0.25, 
             recurrent_dropout = 0.3,return_sequences=True))
    model.add(LSTM(60, kernel_initializer=glorot_uniform(seed=2021),return_sequences=True))
    model.add(LSTM(40, kernel_initializer=glorot_uniform(seed=2021)))
    model.add(Dropout(0.25))
    model.add(Dense(self.n_out, 
            kernel_initializer = glorot_uniform(seed=2021)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer = "RMSprop", metrics = ['categorical_accuracy'])
    return model

  # 学習
  def train(self, x_train, t_train, batch_size, epochs) :
    early_stopping = EarlyStopping(patience=0, verbose=1)
    model = self.create_model()
    model.fit(x_train, t_train, batch_size = batch_size, epochs = epochs, verbose = 1,
          shuffle = True, callbacks = [early_stopping], validation_split = 0.1)
    return model


n_hidden = 70     # 出力次元
epochs = 10      # エポック数
batch_size = 10   # ミニバッチサイズ

# モデル定義
prediction = Prediction(maxlen, n_hidden, n_in, n_out)
# 学習
model = prediction.train(x_train, t_train, batch_size, epochs)
# テスト
score = model.evaluate(x_test, t_test, batch_size = batch_size, verbose = 1)
print("score:", score)

# 正答率、準正答率（騰落）集計
preds = model.predict(x_test)
correct = 0
semi_correct = 0
for i in range(len(preds)):
  pred = np.argmax(preds[i,:])
  tar = np.argmax(t_test[i,:])
  if pred == tar :
    correct += 1
  else :
    if pred+tar == 1 or pred+tar == 5 :
      semi_correct += 1

print("正答率:", 1.0 * correct / len(preds))
print("準正答率（騰落）:", 1.0 * (correct+semi_correct) / len(preds))


