import tensorflow as tf
import numpy as np


sess = None
y_pred = None
x = None

def create_session():
    global sess
    global y_pred
    global x


    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('training_model/my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('training_model/'))
    all_vars = tf.get_collection('vars')


    for v in all_vars:
        v_ = sess.run(v)
        print(v_)

    x = tf.placeholder('float')
    y_pred = tf.add(tf.matmul(x, all_vars[0]), all_vars[1])
    y_pred = tf.nn.softmax(y_pred)


create_session()


def predict(image):
    y = sess.run(y_pred, feed_dict={x: np.reshape(image, [1, 784])})
    print y