import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# mnist.test contains 10K images + labels
# mnist.train contains 60K images + labels
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input image
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# labels (one-hot encoding)
Y_ = tf.placeholder(tf.float32, [None, 10])

# number of neurons in the sigmoid layer
N = 200

# weight and biases of the sigmoid layer
W2 = tf.get_variable('W2', shape=[784, N],
    initializer=tf.truncated_normal_initializer(stddev=0.1))
b2 = tf.get_variable('b2', shape=[N],
    initializer=tf.constant_initializer(0.0))

# weight and biases of the softmax layer
W3 = tf.get_variable('W3', shape=[N, 10],
    initializer=tf.truncated_normal_initializer(stddev=0.1))
b3 = tf.get_variable('b3', shape=[10],
    initializer=tf.constant_initializer(0.0))

# flatten the images
Y1 = tf.reshape(X, [-1, 784])

# sigmoid layer
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
# softmax layer
Y = tf.nn.softmax(tf.matmul(Y2, W3) + b3)

# Loss function
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000

# performance
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# tensorboard
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cross_entropy', cross_entropy)
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('summary/sigmoid-train', sess.graph)
test_writer = tf.summary.FileWriter('summary/sigmoid-test')

# training loop
max_test_accuracy = 0.0
for step in range(20001):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    if step % 50 == 0:
        train_summary, train_accuracy, train_cross_entropy = sess.run([merged, accuracy, cross_entropy],
            feed_dict={X: batch_X, Y_: batch_Y})
        train_writer.add_summary(train_summary, step)

        test_summary, test_accuracy, test_cross_entropy = sess.run([merged, accuracy, cross_entropy],
            feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        test_writer.add_summary(test_summary, step)

        print('step: {}'.format(step))
        print('\ttrain_accuracy: {:.4f}\ttrain_cross_entropy: {:.4f}'.format(train_accuracy, train_cross_entropy))
        print('\ttest_accuracy: {:.4f}\ttest_cross_entropy: {:.4f}'.format(test_accuracy, test_cross_entropy))

        max_test_accuracy = max(max_test_accuracy, test_accuracy)

print('max_test_accuracy: {:.4f}'.format(max_test_accuracy))
