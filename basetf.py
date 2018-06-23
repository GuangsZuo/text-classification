"""
a base tf model class to reuse
"""
import tensorflow as tf

class basetf:
    def __init__(self, learning_rate):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32, trainable=False,
                                           initializer=tf.constant_initializer(1))

    def init_session(self):
        self.sess = tf.Session(config=self.config)
        self.saver = tf.train.Saver()

    def close_session(self):
        self.sess.close()


