import tensorflow as tf
import os
import numpy as np

# fw = open("parameters.txt", "w")


def load_data(data_dir):
    train_digits = []
    train_labels = []
    test_digits = []
    test_labels = []
    for file_name in os.listdir(data_dir):
        if "train" in file_name:
            if "digits" in file_name:
                for line in open(data_dir+file_name):
                    train_digits.append([float(pixel) for pixel in line.strip().split(",")])
            elif "labels" in file_name:
                for line in open(data_dir+file_name):
                    train_labels.append([float(pixel) for pixel in line.strip().split(",")])
            print "training data loaded"
        if "test" in file_name:
            if "digits" in file_name:
                for line in open(data_dir+file_name):
                    test_digits.append([float(pixel) for pixel in line.strip().split(",")])
            elif "labels" in file_name:
                for line in open(data_dir+file_name):
                    test_labels.append([float(pixel) for pixel in line.strip().split(",")])
            print "testing data loaded"
    return test_digits, test_labels, train_digits, train_labels

test_digits, test_labels, train_digits, train_labels = load_data("/Users/reverie-pc/Desktop/library/deep_learning/mnist/data/")

print "data loaded"

image_size = 28
flat_image_size = 28 * 28
num_classes = 10

x = tf.placeholder(tf.float32, [None, flat_image_size])
y_real = tf.placeholder(tf.float32, [None, num_classes])
y_real_cls = tf.argmax(y_real, 1)

weights = tf.Variable(tf.zeros([flat_image_size, num_classes]), name="w")
tf.add_to_collection("vars", weights)
biases = tf.Variable(tf.zeros([num_classes]), name="b")
tf.add_to_collection("vars", biases)

saver = tf.train.Saver()

logits = tf.add(tf.matmul(x, weights), biases)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_real)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_real, y_pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

n_epochs = 10

for epoch in range(n_epochs):
    for i in range(600):
        train_digits_batch = train_digits[i*100: ((i+1)*100)]
        train_labels_batch = train_labels[i*100: ((i+1)*100)]
        train_digits_batch = np.matrix(train_digits_batch)
        train_labels_batch = np.matrix(train_labels_batch)
        session.run(optimizer, feed_dict={x: train_digits_batch, y_real : train_labels_batch})

    # print np.shape(np.matrix(test_digits)), np.shape(np.matrix(test_labels))
    acc = session.run(accuracy, feed_dict={x: np.matrix(test_digits), y_real: np.matrix(test_labels)})
    print acc, "iteration : ", epoch+1

saver.save(session, "my-model")
