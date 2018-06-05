# -*- coding: utf-8 -*-

import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
import mxnet as mx
drop_prob1 = 0.2
drop_prob2 = 0.5

net = nn.Sequential()
# net.add(nn.Flatten())
net.add(nn.Dense(5, activation="relu"))
net.add(nn.Dropout(drop_prob1))
#net.add(nn.Dense(5, activation="relu"))
#net.add(nn.Dropout(drop_prob2))
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))


loss = gloss.L2Loss()
#trainer = gluon.Trainer(net.collect_params(), optimizer=mx.optimizer.RMSProp())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

def read_data(train_data):
    dataset = np.loadtxt(train_data,delimiter='\t')
    X = dataset[:,:-1]
    y = dataset[:,-1]
    return nd.array(X,dtype='float32'),nd.array(y,dtype='float32')

batch_size = 32
num_epochs = 10

# data reader
X,y = read_data('vi.txt')
#print(X.dtype)
dataset = gdata.ArrayDataset(X, y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
        #print(net[0].weight.data())
        print("epoch %d, loss: %f"
                  % (epoch, loss(net(X), y).mean().asnumpy()))
