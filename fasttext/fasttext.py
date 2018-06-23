import tensorflow as tf
import numpy as np

from tqdm import tqdm

from basetf import basetf


class fasttext(basetf):
    def __init__(self, input_size=300, labels_size=1, hidden_units=10, learning_rate=1e-3, batch_size=128, 
                epoch=100, early_stopping_rounds=3):
        super(fasttext,self).__init__(learning_rate)
        self.hidden_units = hidden_units
        self.labels_size = labels_size
        self.input = tf.placeholder(dtype=tf.int32, shape=(None, input_size), name="input")
        self.labels = tf.placeholder(dtype=tf.float64, shape=(None, labels_size), name="target_labels")
        self.batch_size = batch_size
        self.epoch = epoch
        self.early_stopping_rounds = early_stopping_rounds
        self.model_file="./best_model.hdf5"

    def load_embeding(self, embeding, vocab_size=100000, embed_size=300):
        self.embeding = tf.get_variable("embeding",shape=(vocab_size, embed_size), dtype=tf.float64,
                                        initializer=tf.constant_initializer(embeding, verify_shape=True), trainable=False)


    def define_model(self):
        assert self.__getattribute__("embeding") != None
        self.embed = tf.nn.embedding_lookup(self.embeding, self.input) # shape is (batch_size, input_size, embed_size)
        self.embed_mean = tf.reduce_mean(self.embed, axis=1) # shape is (batch_size, embed_size)
#         self.hidden_layer = tf.layers.dense(self.embed_mean, units=self.hidden_units,
#                                             activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.logits = tf.layers.dense(self.embed_mean, units=self.labels_size)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits= self.logits) # shape is (batch_size, 1)
        self.batch_loss = tf.reduce_mean(self.losses)
        self.output = tf.sigmoid(self.logits)

        self.trainop = tf.contrib.layers.optimize_loss(self.batch_loss, self.global_step, None,
                             tf.train.AdamOptimizer(self.learning_rate),summaries=["gradients","loss"])
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./tflogs/", self.sess.graph)

    def train(self, x_train, y_train, x_val, y_val):
        self.sess.run(tf.global_variables_initializer())
        self.best_score = 1
        self.bad_rounds = 0
        for epoch in range(1, self.epoch + 1):
            train_loss = 0
            counter = 0
            for st, ed in tqdm(zip(range(0, x_train.shape[0], self.batch_size),
                                   range(self.batch_size, x_train.shape[0] + self.batch_size, self.batch_size))):
                if ed > x_train.shape[0]: ed = x_train.shape[0]
                x_train_batch = x_train[st:ed]
                y_train_batch = y_train[st:ed]
                losses, _, summary, step= self.sess.run([self.losses, self.trainop, self.merged, self.global_step],
                    feed_dict={self.input: x_train_batch, self.labels: y_train_batch})
                train_loss += np.sum(losses)
                counter += 1
                self.writer.add_summary(summary, step)
            train_loss = train_loss / x_train.shape[0] / self.labels_size
            val_loss = self.evaluate(x_val, y_val)
            print("Epoch %d: train loss : %.6f, val loss: %.6f" % (epoch, train_loss, val_loss))
            if val_loss <= self.best_score:
                print("*** New best score ***\n")
                self.best_score = val_loss
                self.bad_rounds = 0
                self.saver.save(self.sess, self.model_file)
            else:
                self.bad_rounds += 1
                if self.bad_rounds >= self.early_stopping_rounds:
                    print("Epoch %05d: early stopping, best score = %.6f" % (epoch, self.best_score))
                    break

    def evaluate(self, x_val, y_val, batch_size=128):
        val_loss = 0
        for st, ed in zip(range(0, x_val.shape[0], batch_size),
                          range(batch_size, x_val.shape[0] + batch_size, batch_size)):
            x_val_batch = x_val[st:ed]
            y_val_batch = y_val[st:ed]
            losses, _ = self.sess.run([self.losses, self.trainop],
                                      feed_dict={self.input: x_val_batch, self.labels: y_val_batch})
            val_loss += np.sum(losses)
        val_loss = val_loss / x_val.shape[0] / self.labels_size
        return val_loss

    def predict(self, test_data, batch_size=256):
        self.saver.restore(self.sess, self.model_file)
        result = np.zeros((test_data.shape[0], self.labels_size))
        for st, ed in zip(range(0, test_data.shape[0], batch_size),
                          range(batch_size, test_data.shape[0] + batch_size, batch_size)):
            if ed > test_data.shape[0]: ed = test_data.shape[0]
            x_val_batch = test_data[st:ed]
            result[st:ed] = self.sess.run(self.output, feed_dict={self.input: x_val_batch})             
        return result