import cifar100
import tensorflow as tf
import os

def conv2d(input, filter):
    """
    Description
    -----------
    Simplify the process of creating a conv2d by always using a stride of 1 and SAME padding
    
    Parameters
    ----------
    input: tensor, 4D image with shape (Batch_size, Width, Height, Channels)
        image to convolve
    filter: tensor, 4D filter with shape (Width, Height, In_Channels, Out_Channels) 
        filter to apply to the image
        In_Channels are the number of channels of the input for which the filter will convolve
        Out_Channels are the number of channels the user wishes to output for the next 'image'
    
    Returns
    -------
    a 4D tensor after the convolve operation has been applied to the input image
    """
    return tf.nn.conv2d(input = input, filter = filter, strides = [1, 1, 1, 1], padding = 'SAME')

# If using spyder, set to True. Need to always reset the default graph so that the old one is removed
spyder = True
if spyder:
    tf.reset_default_graph()


image_format = 1 # 0 == pickle file, 1 == binary file, 2 == reading multiple image(PNG) files

if(image_format == 0):
    # reading pickle file
    image_paths = ['./cifar-100-python/train']
    file_path = './cifar-100-python/test'
elif(image_format == 1):
    # reading binary file
    image_paths = ['./cifar-100-binary/train.bin']
    file_path = './cifar-100-binary/test.bin'
elif(image_format == 2):
    # reading multiple image(PNG) files
    images_paths = tf.train.match_filenames_once('./train/*.png')
    # file_path = tf.train.match_filenames_once('./test/*.png')
    
total_images = 50000

# Hyperparameters for the neural network
num_epochs = 1000
learning_rate = 0.01
keep_prob = 0.6
batch_size = 64

# Image specifications (Cifar-100 data is composed of 32x32x3 images)
image_height = 32
image_width = 32
image_depth = 3

num_classes = 100

cifar101 = cifar100.Cifar100(image_format = image_format, image_paths = image_paths, total_images = total_images, 
                             batch_size = batch_size, num_epochs = num_epochs)

images_test, labels_test = cifar101.test_data(file_path)

if(image_format == 1 or image_format == 2):
    images_train, labels_train = cifar101.next_batch()

training = tf.placeholder(dtype = tf.bool, shape = [])

global_step = tf.Variable(0, trainable = False)

inputs = tf.placeholder(dtype = tf.float32, shape = [None, image_height, image_width, image_depth])
labels = tf.placeholder(dtype = tf.int32, shape = [None])

with tf.device('/device:GPU:0'):
    # 3x3x3 filter with 16 out channels
    # Conv(Same padding)->Relu->BN->Dropout
    with tf.variable_scope('Conv1'):
        filter1 = tf.get_variable('filter', shape = [3, 3, 3, 16], initializer = tf.contrib.layers.xavier_initializer())
        layer1 = conv2d(inputs, filter1)
        layer1 = tf.nn.relu(layer1)
        layer1 = tf.contrib.layers.batch_norm(layer1)
        layer1 = tf.contrib.layers.dropout(layer1, keep_prob = keep_prob, is_training = training)

    # 3x3x16 filter with 32 out channels
    # Conv(Same padding)->Relu->BN->Dropout
    with tf.variable_scope('Conv2'):
        filter2 = tf.get_variable('filter', shape = [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer())
        layer2 = conv2d(layer1, filter2)
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.contrib.layers.batch_norm(layer2)
        layer2 = tf.contrib.layers.dropout(layer2, keep_prob = keep_prob, is_training = training)
       
    # 3x3x32 filter with 32 out channels
    # Conv(Same padding)->Relu->Max_pool->BN->Dropout
    with tf.variable_scope('Conv3'):
        filter3 = tf.get_variable('filter', shape = [3, 3, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
        layer3 = conv2d(layer2, filter3)
        layer3 = tf.nn.relu(layer3)
        layer3 = tf.nn.max_pool(layer3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        layer3 = tf.contrib.layers.batch_norm(layer3)
        layer3 = tf.contrib.layers.dropout(layer3, keep_prob = keep_prob, is_training = training)

    # 3x3x32 filter with 64 out channels
    # Conv(Same padding)->Relu->Max_pool->BN->Dropout
    with tf.variable_scope('Conv4'):
        filter4 = tf.get_variable('filter', shape = [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
        layer4 = conv2d(layer3, filter4)
        layer4 = tf.nn.relu(layer4)
        layer4 = tf.contrib.layers.batch_norm(layer4)
        layer4 = tf.contrib.layers.dropout(layer4, keep_prob = keep_prob, is_training = training)

    # 3x3x64 filter with 64 out channels
    # Conv(Same padding)->Relu->Max_pool->BN->Dropout
    with tf.variable_scope('Conv5'):
        filter5 = tf.get_variable('filter', shape = [3, 3, 64, 64], initializer = tf.contrib.layers.xavier_initializer())
        layer5 = conv2d(layer4, filter5)
        layer5 = tf.nn.relu(layer5)
        layer5 = tf.contrib.layers.batch_norm(layer5)
        layer5 = tf.contrib.layers.dropout(layer5, keep_prob = keep_prob, is_training = training)

    # Flatten Conv into 16x16x64 FC
    flat = tf.contrib.layers.flatten(layer5)

    # 16384 x 4096 FC
    # FC->BN->Dropout
    with tf.variable_scope('FC6'):
        layer6 = tf.contrib.layers.fully_connected(flat, 4096)
        layer6 = tf.contrib.layers.batch_norm(layer6)
        layer6 = tf.contrib.layers.dropout(layer6, keep_prob = keep_prob, is_training = training)
        
    # 4096 x 4096 FC
    # FC->BN->Dropout
    with tf.variable_scope('FC7'):
        layer7 = tf.contrib.layers.fully_connected(layer6, 4096)
        layer7 = tf.contrib.layers.batch_norm(layer7)
        layer7 = tf.contrib.layers.dropout(layer7, keep_prob = keep_prob, is_training = training)
    
    # 4096 x 1024 FC
    # FC->BN->Dropout
    with tf.variable_scope('FC8'):
        layer8 = tf.contrib.layers.fully_connected(layer7, 1024)
        layer8 = tf.contrib.layers.batch_norm(layer8)
        layer8 = tf.contrib.layers.dropout(layer8, keep_prob = keep_prob, is_training = training)
    
    logits = tf.contrib.layers.fully_connected(layer8, num_classes)

one_hot_label = tf.one_hot(labels, depth = num_classes, axis = 1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_label))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step) 

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    if os.path.exists('tmp/cifar_100.ckpt.index'): 
        saver.restore(sess, "tmp/cifar_100.ckpt")
        print("Model restored.")
    
    # session running for pickle images
    if(image_format == 0):
        while not cifar101.complete():
            train_images, train_labels = cifar101.next_batch()
            _, error = sess.run([optimizer, loss], feed_dict = {training: True, inputs: train_images, labels: train_labels})
            print('Step %i...' % (global_step.eval()))
            print('Error of %f' % (error))
            
            if (global_step.eval() % 10 == 0):
                print('\tTest Time!!!!!')
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_label, 1)), dtype = tf.float32))
                acc = sess.run(accuracy, feed_dict = {training: False, inputs:images_test, labels:labels_test})
                print('\t\t\tAccuracy of %f' % (acc))   
            
     # session running for binary and PNG images       
    else:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        
        try:
            while not coord.should_stop():    
                py_images, py_labels = sess.run([images_train, labels_train])
                _, error = sess.run([optimizer, loss], feed_dict = {training: True, inputs: py_images, labels: py_labels})
                print('Step %i...' % (global_step.eval()))
                print('Error of: %f' % (error))
                    
                if (global_step.eval() % 10 == 0):
                    print('\tTest Time!!!!!')
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_label, 1)), dtype = tf.float32))
                    acc = sess.run(accuracy, feed_dict = {training: False, inputs:images_test, labels:labels_test})
                    print('\t\t\tAccuracy of %f' % (acc))                
        
        # On spyder, termination via keyboard interrupt will not catch the exception hence the checkpoint will not be saved.
        except: #Exception as ex:
            #print(ex)
            print('Done Training ------- Epoch limit reached!')
    
        coord.request_stop()
        coord.join(threads)
    
    save_path = saver.save(sess, "tmp/cifar_100.ckpt")
    print("Model saved in file: %s" % (save_path))