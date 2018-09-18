import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("fsd")