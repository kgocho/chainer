# -*- coding: utf-8 -*-
'''
Created on 2015/11/13

@author: gocho
'''
"""
chainerを使ったcifar-10の画像データの分析。cifar-10の画像は32*32=1024。RGBの順に1024×3=3072次元のデータ。0～255
モデルはほぼMNISTと同じ
train mean loss: 0.403807
test accuracy: 0.720800
time: 11522.0335546
これくらいの精度
"""
import chainer
import chainer.functions as F
from chainer import optimizers

import numpy as np
from pylab import *
import time


#学習のパラメータ
batchsize = 100
n_epoch = 20


"データセットの準備"
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

x_train = None
y_train = []

for i in xrange(1,6):
    data_dictionary = unpickle("cifar-10-batches-py/data_batch_%d" % i)
    if x_train == None:
        x_train = data_dictionary['data']
    else:
        x_train = np.vstack((x_train,data_dictionary['data']))
    y_train = y_train + data_dictionary['labels']

test_data_dictionary = unpickle("cifar-10-batches-py/test_batch")
x_test = test_data_dictionary['data']

y_train = np.array(y_train)
y_test = np.array(test_data_dictionary['labels'])

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_train /= 255.0
x_test /= 255.0


# 画像を (nsample, channel, height, width) の4次元テンソルに変換
x_train = x_train.reshape((len(x_train),3, 32, 32))
x_test = x_test.reshape(len(x_test),3,32,32)

#訓練データの個数
N = 50000
N_test = y_test.size
#モデル定義
model = chainer.FunctionSet(
                            conv1=F.Convolution2D(3, 20, ksize=5),
                            conv2=F.Convolution2D(20, 50, ksize=5),
                            l1=F.Linear(1250, 500),
                            l2=F.Linear(500, 10)
                            )


#順伝播の定義
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    #1層目の畳み込みの後にプーリング層
    h1 = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    #2層目の畳み込みの後にプーリング層
    h2 = F.max_pooling_2d(F.relu(model.conv2(h1)), 2)
    #3層目の出力
    h3 = F.dropout(F.relu(model.l1(h2)), train=train)
    #yの出力
    y = model.l2(h3)

    # 訓練時とテスト時で返す値を変える
    if train:
        # 訓練時は損失を返す
        # 多値分類なのでクロスエントロピーを使う
        loss = F.softmax_cross_entropy(y, t)
        return loss
    else:
        # テスト時は精度を返す
        acc = F.accuracy(y, t)
        return acc

# Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
optimizer.setup(model)

#誤差と精度を書くためのファイルを用意
fp1 = open("accuracy_cifar.txt", "w")
fp2 = open("loss_cifar.txt", "w")

fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")


#訓練ループ
#各エポックでテスト精度を求める
start_time = time.clock()
for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch
    nowtime = time.clock()
    print "elapsed time", nowtime - start_time

    #訓練データを用いてパラメータを更新する
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = np.asarray(x_train[perm[i:i + batchsize]])
        y_batch = np.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(y_batch)

    print "train mean loss: %f" % (sum_loss / N)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()

    #テストデータを用いて精度を評価する
    sum_accuracy = 0
    for i in range(0, N_test, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])
        y_batch = np.asarray(y_test[i:i + batchsize])

        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)

    print "test accuracy: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()

end_time = time.clock()
print "time:" , end_time - start_time

fp1.close()
fp2.close()

#学習したモデルを保存する
import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
#model.to_cpu()
cPickle.dump(model, open("model-cifar-10.pkl", "wb"), -1)
