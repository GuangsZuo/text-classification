import tensorflow as tf
import numpy as np

from tqdm import tqdm

from .. import basetf


class fasttext(basetf):
    def __init__(self, input_size=300, labels_size=1, hidden_units=10, learning_rate=1e-3, batch_size=128, classes=2):
        super(fasttext,self).__init__(learning_rate)
        self.hidden_units = hidden_units
        self.labels_size = labels_size
        self.input = tf.placeholder(dtype=tf.int32, shape=(None, input_size), name="input")
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, labels_size), name="target_labels")
        self.classees = classes
        self.batch_size = batch_size

    def load_embeding(self, embeding, vocab_size=100000, embed_size=300):
        self.embeding = tf.get_variable(shape=(vocab_size, embed_size), dtype=tf.float64,
                                        initializer=tf.constant_initializer(embeding, verify_shape=True))


    def define_model(self):
        assert self.__getattribute__("embeding") != None
        self.embed = tf.nn.embedding_lookup(self.embeding, self.input) # shape is (batch_size, input_size, embed_size)
        self.embed_mean = tf.reduce_mean(self.embed, axis=1) # shape is (batch_size, embed_size)
        self.hidden_layer = tf.layers.dense(self.embed_mean, units=self.hidden_units,
                                            activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.logits = tf.layers.dense(self.hidden_layer, units=self.labels_size)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits= self.logits) # shape is (batch_size, 1)
        self.batch_loss = tf.reduce_mean(self.losses)

        self.trainop = tf.contrib.layers.optimize_loss(self.batch_loss, self.global_step, None,
                                                       tf.train.AdamOptimizer(self.learning_rate))

    def model_train(self, x_train, y_train, x_val, y_val):

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
                losses, _= self.sess.run([self.losses, self.trainop],
                    feed_dict={self.input: x_train_batch, self.labels: y_train_batch})
                train_loss += np.sum(losses)
                counter += 1
            train_loss = train_loss / x_train.shape[0] / self.classes
            val_loss = self.evaluate_model(x_val, y_val)
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

    def evaluate_model(self, x_val, y_val, batch_size=128):
        val_loss = 0
        for st, ed in zip(range(0, x_val.shape[0], batch_size),
                          range(batch_size, x_val.shape[0] + batch_size, batch_size)):
            x_val_batch = x_val[st:ed]
            y_val_batch = y_val[st:ed]
            losses, _ = self.sess.run([self.model.losses, self.model.trainop],
                                      feed_dict={self.model.input: x_val_batch, self.model.target: y_val_batch})
            val_loss += np.sum(losses)
        val_loss = val_loss / x_val.shape[0] / self.classes
        return val_loss

    def split_train_data(self, data):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in skf.split(data[0], data[1]):
            return data[0][train_index], data[1][train_index], data[0][val_index], data[1][val_index]

    def model_predict(self, test_data, batch_size=1024):
        self.saver.restore(self.sess, self.model_file)
        result = np.zeros((test_data.shape[0], self.classes))
        for st, ed in zip(range(0, test_data.shape[0], batch_size),
                          range(batch_size, test_data.shape[0] + batch_size, batch_size)):
            if ed > test_data.shape[0]: ed = test_data.shape[0]
            x_val_batch = test_data[st:ed]
            output = self.sess.run([self.model.output, ], feed_dict={self.model.input: x_val_batch})
            print(type(output[0]))
            result[st:ed] = output[0]
        return result





