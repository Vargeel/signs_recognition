import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas

input_size = 64

sess = tf.InteractiveSession()



# I initially did my own training/testing split of the data because I did not notice that you provided one. I left the code in as it is an interesting aspect of data separation.

#get image scores, these are the expected value of the score for each image
# dat=pandas.read_csv('data.csv')

# names=dat['image_paths'].values
# scores=np.load('targets.npy')

#Get image data
# images=np.zeros((len(names),input_size,input_size,3))
# for i in range(0,len(names)):
#     im = Image.open(names[i])
#     images[i,:,:,:] =im


# r = np.random.permutation(len(names))
#
# images = images[r,:,:,:]
# scores = scores[r]

# Split training and test sets and normalize test data
# nb_images = len(names)
# test_set_nb = 200
# train_set_nb = nb_images - test_set_nb
#
#
# images=images/255 # normalize data
# images_train = images[0:train_set_nb,:,:,:] #training set
# images_test = images[train_set_nb+1:nb_images,:,:,:] #test set
# #image_train_mean = np.mean(images_train,dtype=np.float32)
# image_train_mean = 0
# images_train = images_train-image_train_mean
# images_test = images_test-image_train_mean
# del images
#
#
# labels_train = np.asarray(scores[0:train_set_nb]) #training set
# labels_test = np.asarray(scores[train_set_nb+1:nb_images]) #test set
#



# Here I split the data using the lists already given

root = '../'
labels_train = []
images_path_train = []
labels_test = []
images_path_test = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if name=='test.txt':
            f = open(os.path.join(path, name))
            list_test = f.read().split('\n')
            list_test.remove('')
            for idx in range(len(list_test)):
                images_path_test.append(os.path.join(path, list_test[idx]))
                target = np.zeros(10)
                target[int(path[-5:])-1] = 1
                labels_test.append(target)


        if name=='train.txt':
            f = open(os.path.join(path, name))
            list_train = f.read().split('\n')
            list_train.remove('')
            for idx in range(len(list_train)):
                images_path_train.append(os.path.join(path, list_train[idx]))
                target = np.zeros(10)
                target[int(path[-5:])-1] = 1
                labels_train.append(target)

#Get image data in variables
images=np.zeros((len(images_path_train),input_size,input_size,3))
for i in range(0,len(images_path_train)):
    im = Image.open(images_path_train[i])
    images[i,:,:,:] =im
r = np.random.permutation(len(images_path_train))
images_train = images[r,:,:,:]
labels_train = np.asarray(labels_train)[r]
# the training data is now shuffled

images_t=np.zeros((len(images_path_test),input_size,input_size,3))
for i in range(0,len(images_path_test)):
    im = Image.open(images_path_test[i])
    images_t[i,:,:,:] =im
images_test = images_t
labels_testn = np.asarray(labels_test)

train_set_nb = len(images_path_train)


# Network utilities
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def weight_constant(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def bias_constant(shape):
    initial = tf.constant(0.01, shape=shape)
    return initial

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def mean_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def getBatch(mode,bsize,images_train,images_test,labels_train,labels_test):
    batch = [0]*2
    batch[0] = np.zeros((bsize,input_size,input_size,3))
    batch[1] = np.zeros((bsize,1))
    if mode == 'train':
        r = np.random.randint(train_set_nb,size=(bsize))
        batch[0] = images_train[r,:,:,:]
        rr = np.random.randint(2)
        if rr == 1:
            batch[0] = np.transpose(batch[0],(1,2,3,0))
            batch[0] = np.fliplr(batch[0])
            batch[0] = np.transpose(batch[0],(3,0,1,2))

        batch[1] = labels_train[r]
    if mode == 'test':
        batch[0] = images_test
        batch[1] = labels_test
    return batch
    if mode == 'test_out':
        batch[0] = images_test
        batch[1] = np.expand_dims(np.zeros(len(images_test)),1)
    return batch


# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, input_size,input_size,3])
y = tf.placeholder(tf.float32, shape=[None, 10])


# Network 3 convolution layers followed by 2 fully connected ones
W_conv1 = weight_variable([3, 3, 3, 32])  #64
b_conv1 = weight_variable([32])
h_conv1 = tf.nn.elu( conv2d(x, W_conv1) + b_conv1  )

h_pool1 = max_pool_2x2(h_conv1)#64

W_conv2 = weight_variable([3, 3, 32, 64])  #32
b_conv2 = weight_variable([64])
h_conv2 = (tf.nn.elu( conv2d(h_pool1, W_conv2) + b_conv2) )

h_pool2 = max_pool_2x2(h_conv2)  #16

W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.elu( conv2d(h_pool2, W_conv4) + b_conv4 )

h_pool5 = max_pool_2x2(h_conv4)  #4

W_fc1 = weight_variable([8*8*128, 256])
b_fc1 = bias_variable([256])
h_conv6_flat = tf.reshape(h_pool5, [-1, 8*8*128])

h_fc1 = tf.nn.elu(tf.matmul(h_conv6_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Optimize for MSE with Adam
l2_factor = 0.0001
l2_loss = l2_factor * tf.nn.l2_loss(W_fc1, name=None)  #l2 regularization

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv)) + l2_loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# Train network
bsize=256
converged=0
prev_test_err=99999
iterator_training=0
max_test_accuracy = 0.90
best_epoch = 0
while converged==0:
    iterator_training +=1
    batch=getBatch('train',bsize,images_train,images_test,labels_train,labels_test)
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.50})
    if iterator_training%20 == 0: #Get training error
        accuracy_train = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
        print "Training accuracy: {}".format(accuracy_train)
    if iterator_training%20 == 0: #Get test error
        batch = getBatch('test',bsize,images_train,images_test,labels_train,labels_test)
        accuracy_test = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
        if accuracy_test > max_test_accuracy:
            max_test_accuracy = accuracy_test
            best_epoch = iterator_training
            if iterator_training > 10 :

                print 'best result : '
                print max_test_accuracy
                print 'at epoch :'
                print best_epoch

        if iterator_training > 800:
            converged = 1

        print "Validation accuracy: {}\n".format(accuracy_test)
        print iterator_training

