# -*- coding: utf-8 -*-
'''
Created on 2015/10/15

@author: gocho
'''
import chainer
import chainer.functions as F
from chainer import optimizers

import numpy as np
from pylab import *
import time


"""
モデル定義
"""

unit=3
model=chainer.FunctionSet(
                          l1=F.Linear(1,unit),     #入力層ー隠れ層
                          l2=F.Linear(unit,1)      #隠れ層ー出力層
                          )

#重み更新の定義
opt=optimizers.SGD(lr=0.01)              #重みの更新は勾配降下法を用いる。デフォルトの学習率はlr=0.01
opt.setup(model)

print model.l1.W
print model.l2.W
"""
順伝播
"""
def forward(x_train, t_train):
    x, t =chainer.Variable(x_train), chainer.Variable(t_train)      #訓練データをchainerで使える方に変換
    h1=F.tanh(model.l1(x))          #tanh関数
    #h1=F.sigmoid(model.l1(x))       #sigmoid関数
    #h1=F.relu(model.l1(x))          #relu関数
    y=model.l2(h1)
    loss = F.mean_squared_error(y, t)
    return loss, y          #平均2乗誤差と出力を返す

"""
訓練データ作成
"""
def heviside(x):
    """Heviside関数"""
    return 0.5 * (np.sign(x) + 1)

def make_data():
    x=np.linspace(-1,1,50)      #-1から1まで50分割
    t=np.square(x)      #t=x^2
    #t=np.sin(x)         #t=sin(x)
    #t=np.abs(x)         #t=|x|
    #t=heviside(x)         #t=heviside(x)
    return x.astype(np.float32), t.astype(np.float32)       #chainerではfloat32型でないとエラー。numpyのデフォルトはfloat64型

"""
学習。逐次学習で訓練データを1つ1つ見ていくが、順番はランダム
"""

def learn(x_train,t_train,epoch=2000,epsilon=0.0001):
    i=0
    while (i<epoch):
        perm = np.random.permutation(len(x_train))               #0～訓練データ数までランダムに配置した配列
        sum_loss=0
        for j in range(len(x_train)):                   #訓練データの個数分ループ
            x = np.array(x_train[perm[j]]).reshape(1,1)         #訓練データをランダムに並べ替えて使う
            t = np.array(t_train[perm[j]]).reshape(1,1)
            opt.zero_grads()
            loss,y=forward(x, t)
            loss.backward()
            opt.update()
            sum_loss += loss.data
        train_mean_loss = sum_loss/len(x_train)
        if(train_mean_loss<epsilon):                #誤差がepsilon以下なら終了
            print "finish epoch =" , i
            print "train mean loss =", sum_loss/len(x_train)
            break
        if(i%100==0):
            print "epoch=", i
            print "train mean loss =", sum_loss/len(x_train)
        i+=1

x_train, t_train = make_data()
x_train,y_train = x_train.reshape(len(x_train),1),t_train.reshape(len(t_train),1)

start_time = time.clock()

learn(x_train, y_train,epoch=10000)#5000

end_time = time.clock()
print "time:", end_time - start_time

loss, t_test=forward(x_train, y_train)
print "test mean loss", loss.data

#訓練データ
plot(x_train,t_train,'bo')

#学習結果
plot(x_train,t_test.data,'r-')

#隠れ層のうニットの表示
hidden=F.tanh(model.l1(chainer.Variable(x_train))).data
plot(x_train, hidden[:,0], 'y--')
plot(x_train, hidden[:,1], 'g--')
plot(x_train, hidden[:,2], 'm--')

ylim([0, 1])
#重みベクトルの表示
print model.l1.W
print model.l1.b
print model.l2.W
print model.l2.b

show()
