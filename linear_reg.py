# -*- coding: utf-8 -*-
from mxnet import nd,autograd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

# create dataset 
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)


# data reader
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# create model
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))

# loss
loss = gloss.L2Loss()

# trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# train
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print("epoch %d, loss: %f"
          % (epoch, loss(net(features), labels).mean().asnumpy()))

    
dense = net[0]
print('real w',true_w, 
      '\npred w:\n',dense.weight.data()
)
print('real b:',true_b, 
      '\npred b:',dense.bias.data()
)