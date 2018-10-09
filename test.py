import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(tf.__version__)


input1=tf.placeholder(tf.float64)
input2=tf.placeholder(tf.float64)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[7.]}))

