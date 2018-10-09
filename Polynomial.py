import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def model(x,w):
    tmps=[]
    for i in range(parm_num):
        tmp=tf.multiply(w[i],tf.pow(x,i))
        tmps.append(tmp)
    return tf.add_n(tmps)


learning_rate=0.01
training_epoch=50
parm_num=6
w_val=0


train_X=np.linspace(0,100,1001)
train_Y=np.power(train_X,2)*3

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
w=tf.Variable([0.]*parm_num)

y=model(X,w)
cost=(tf.pow(Y-y,2))

train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(tf.clip_by_value(cost,1e-8,1.0))

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(training_epoch):
        for(x,y) in zip(train_X,train_Y):
            sess.run(train_op,feed_dict={X:x,Y:y})

    w_val = sess.run(w)
    print(w_val)

plt.scatter(train_X, train_Y)

train_Y2 = 0
for i in range(parm_num):
    train_Y2 += w_val[i] * np.power(train_X, i)
plt.plot(train_X, train_Y2, 'r')
plt.show()