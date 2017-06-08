import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# mnist.test contains 10K images + labels
# mnist.train contains 60K images + labels
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input image batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# labels (one hot encoding)
Y_ = tf.placeholder(tf.float32, [None, 10])

# number of neurons in the hidden layer
K = 6
L = 12
M = 24
N = 200

W2 = tf.get_variable('W2', shape=[6, 6, 1, K], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2', shape=[K], initializer=tf.constant_initializer(0.0))

W3 = tf.get_variable('W3', shape=[5, 5, K, L], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3', shape=[L], initializer=tf.constant_initializer(0.0))

W4 = tf.get_variable('W4', shape=[4, 4, L, M], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable('b4', shape=[M], initializer=tf.constant_initializer(0.0))

W5 = tf.get_variable('W5', shape=[7 * 7 * M, N], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable('b5', shape=[N], initializer=tf.constant_initializer(0.0))

W6 = tf.get_variable('W6', shape=[N, 10], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.get_variable('b6', shape=[10], initializer=tf.constant_initializer(0.0))


# convolutional layer #2
Y2 = tf.nn.relu(tf.nn.conv2d(X, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
# convolutional layer #3
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + b3)
# convolutional layer #4
Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, 2, 2, 1], padding='SAME') + b4)
# flatten the output from convolitional layer #4
Y4flat = tf.reshape(Y4, shape=[-1, 7 * 7 * M])
# fully-connected rely layer #5
Y5 = tf.nn.relu(tf.matmul(Y4flat, W5) + b5)
# regularization with drop out
pkeep = tf.placeholder(tf.float32, [])
Y5drop = tf.nn.dropout(Y5, pkeep)
# fully-connected softmax later #6
Ylogits = tf.matmul(Y5drop, W6) + b6
Y = tf.nn.softmax(Ylogits)

# Loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# training with learning-rate decay
global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0))
learning_rate = tf.maximum(tf.train.exponential_decay(0.003, global_step, 500, 0.95, staircase=False), 0.0001)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

# performance
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# tensorboard
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cross_entropy', cross_entropy)
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('summary/conv-do-train', sess.graph)
test_writer = tf.summary.FileWriter('summary/conv-do-test')

# training loop
max_test_accuracy = 0.0
for step in range(20000):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})

    if step == 0 or (step + 1) % 50 == 0:
        summary, train_accuracy, train_cross_entropy = sess.run([merged, accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        train_writer.add_summary(summary, step)

        summary, test_accuracy, test_cross_entropy = sess.run([merged, accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        test_writer.add_summary(summary, step)

        print('step: {}'.format(step))
        print('\ttrain_accuracy: {:.4f}\ttrain_cross_entropy: {:.4f}'.format(train_accuracy, train_cross_entropy))
        print('\ttest_accuracy: {:.4f}\ttest_cross_entropy: {:.4f}'.format(test_accuracy, test_cross_entropy))

        max_test_accuracy = max(max_test_accuracy, test_accuracy)

print('max_test_accuracy: {:.4f}'.format(max_test_accuracy))
