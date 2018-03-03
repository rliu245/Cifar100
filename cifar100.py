import tensorflow as tf
import numpy as np
import pickle

class Cifar100: 
    def __init__(self, image_format, image_paths, total_images, batch_size = 32, num_epochs = 1, num_threads = 3, min_after_dequeue = 100):
        """
        Parameters
        ----------
        image_format(required): int
            Must be either 0, 1, or 2. Will raise an exception if the int value isnt 0, 1, or 2.
            0 represents the class will be used for reading a pickle file (python version of the dataset)
            1 represents the class will be used for reading a binary file (binary version of the dataset)
            2 represents the class will be used for reading multiple image files (in the format of JPEG/PNG/etc.)
        image_paths(required): list of strings
            Stores a list where each element is a string of file paths.
            This parameter complements image_format in that image_paths will read image data corresponding to the image_format value
            Example: ['./cifar-100-python/train']
                     ['./cifar-100-python-bin/data_batch_1.bin', './cifar-100-python-bin/data_batch_2.bin', './cifar-100-python-bin/data_batch_3.bin']
                     ['./train/bos_taurus_s_00507.png', './train/stegosaurus_s_000125.png', './train/phone_s_002161.png', './train/squirrel_s_002567.png']
        total_images(required): int
            The total number of images in the training set
        batch_size(default 32): int
            The batch size for which the user wants to process
        num_epochs(default 1): int
            Number of times the user wants to iterate over the whole training set
        num_threads(default 3): int
            The number of threads enqueueing into the batch queue
        min_after_dequeue(default 100): int
            Minimum number of elements in the batch queue after a dequeue; used to ensure mixing of elements in the batch queue 
        """
        self.image_format = image_format
        self.image_height = 32
        self.image_width = 32
        self.image_depth = 3
        self.label_size = 2
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue
        self.capacity = self.min_after_dequeue + self.num_threads * self.batch_size
        self.num_epochs = num_epochs
        self.counter = 0
        
        if(image_format == 0):
            # Unpickle the file and store the data into a dictionary 
            with open(self.image_paths[0], 'rb') as fo:
                self.pickle_data = pickle.load(fo, encoding = 'bytes')
        elif(image_format == 1 or image_format == 2):
            # Create queue to hold all the filenames for the binary files 
            self.filename_queue = tf.train.string_input_producer(self.image_paths, num_epochs = num_epochs)
        
            # Create reader to read from the filename queue 
            self.reader = tf.FixedLengthRecordReader(self.label_size + (self.image_height * self.image_width * self.image_depth))
        else:
            raise Exception("Error in init: image_format isn't 0, 1, or 2. Invalid number!")
        
        # Declare the image and label vars
        self.images = 0
        self.labels = 0
        
        self.shuffle_ordering = np.array([], dtype = np.dtype(int))
        
        self.finished = False
        
        for _ in range(num_epochs):
            temp = np.arange(total_images)
            np.random.shuffle(temp)
        
            self.shuffle_ordering = np.append(self.shuffle_ordering, temp)
    
    def generate_image_files(self, dataset, meta):
        """
        Description
        -----------
        Purpose of this function is to convert the Cifar-100 dictionary dataset into png files
    
        Parameters
        ----------
        dataset: dict
            Dictionary of 5 elements where each element is a byte string
            Expecting elements to be [b'batch_label', b'coarse_labels', b'data', b'filenames', b'fine_labels'] 
        meta: dict
            Dictionary of 2 elements where each element is a byte string    
            Expecting elements to be [b'coarse_label_names', b'fine_label_names']
    
        Returns
        -------
        No return value.
        """
        from PIL import Image
    
        data = dataset[b'data']
        file_names = dataset[b'filenames']
        images = data.reshape([-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i, :, :, :])
            img.save('train/%i'%(i) + file_names[i].decode('utf-8'))

        import csv
        csvfile = './train/labels.csv'

        image_labels = dataset[b'coarse_labels']
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator = ',')
            for val in image_labels:
                writer.writerow([val])
        
    
    def __read_image_binary(self):
        """
        Description
        -----------
        The purpose of this helper function is to read in a binary file and output the image and the fine label.
        
        Returns
        -------
        image
            a 3D tensor (shaped Width, Height, Channels) of images
        fine_label
            a tensor vector of labels
        """
        # Create queue to hold all the filenames for the binary files 
        # self.filename_queue = tf.train.string_input_producer(self.image_paths, num_epochs = self.num_epochs)
        # self.filename_queue = tf.train.string_input_producer(self.image_paths)

        # Calculates how many bytes to read at a time so only 1 image and label is read on each read operation      
        total_bytes = self.label_size + (self.image_height * self.image_width * self.image_depth)
        
        # Read an image and its corresponding label from the filename_queue 
        _, value = self.reader.read(self.filename_queue)
        
        # Data returned from the reader is in byte string format so we must decode it and change the dtype to float32(for image processing later on)
        record_bytes = tf.decode_raw(value, tf.uint8)
        
        # Extract the label (first element is the coarse label. second element is the fine label)
        fine_label = tf.cast(tf.strided_slice(record_bytes, [1], [self.label_size]), tf.int32)
        fine_label = tf.reshape(fine_label, [1])
        
        # Extract the image and since it's read depth first, we must transpose the dimensions
        image = tf.cast(tf.strided_slice(record_bytes, [self.label_size], [total_bytes]), tf.float32)
        image = tf.reshape(image, [self.image_depth, self.image_width, self.image_height])
        image = tf.transpose(image, [1, 2, 0])
        
        return image, fine_label
    
    def __read_image_files(self):
        """
        Description
        -----------
        The purpose of this helper function is to read in image files(specifically PNG format) and output the corresponding image and label
        
        Returns
        -------
        image
            a 3D tensor (shaped Width, Height, Channels) of images
        label
            a tensor vector of labels
        """        
        # Read in 1 image file from the filename_queue
        _, value = self.reader.read(self.filename_queue)
        
        # Data returned from the reader is a string tensor so we must decode it into an image
        image = tf.image.decode_image(value)
        
        # Obtain the label
        label = tf.convert_to_tensor(self.labels[self.counter])
        
        return image, label
    
    def __read_image_dict(self):
        """
        Description
        -----------
        The purpose of this helper function is to read in a pickle file and extract the image data and corresponding labels
        
        Returns
        -------
        image
            a 3D numpy array (shaped Width, Height, Channels) of images
        label
            a numpy vector of labels
        """
        # Extract the images from the dictionary
        self.images = np.array(self.pickle_data[b'data'].reshape([-1, 3, 32, 32]))
        self.images = self.images.transpose([0, 2, 3, 1])
        
        # Extract the labels from the dictionary
        self.labels = np.array(self.pickle_data[b'fine_labels'])
        
        # Shuffle the images and labels
        image = self.images[self.shuffle_ordering[self.counter]]
        
        label = self.labels[self.shuffle_ordering[self.counter]]

        if((self.counter + 1) >= self.shuffle_ordering.size):
            self.counter = 0

            self.finished = True
        else:
            self.counter += 1
        
        return image, label
    
    # Temporary function for determining when we're done training all the epochs for the pickle file format
    def complete(self):
        """
        Description
        -----------
        Only used for when a pickle file is read. Useful for determining whether we are done training all the epochs.
        
        Returns
        -------
        self.finished
            boolean var to determine whether we are done with all the epochs in training phase
        """
        return self.finished
    
    def __read_image(self):
        """
        Description
        -----------
        Helper function to encapsulate the process of reading an image, whether it is a pickle file, binary file, or series of PNG files.
        
        Returns
        -------
        image
            if pickle file is read, returns a 3D numpy array (shaped Width, Height, Channels) of images
            if binary file is read, a 3D tensor (shaped Width, Height, Channels) of images is returned
        label
            if pickle file is read, returns a numpy vector of labels
            if binary file is read, a tensor vector of labels are returned
        
        """
        
        if(self.image_format == 0):
            image, label = self.__read_image_dict()
        elif(self.image_format == 1):
            image, label = self.__read_image_binary()
        elif(self.image_format == 2):
            image, label = self.__read_image_files()
        else:
            raise Exception("Error in read_image func: image_format isn't 0, 1, or 2. Invalid number!")
        
        return image, label
    
    # Way to preprocess an image. An idea for future implementation to enhance the neural network
    '''
    def image_preprocessing(self, image, label, max_delta = 0.4):
        """
        This function is meant to flip an image vertically and horizontally. In addition, it randomly adjusts the brightness to account for different lighting conditions.
        """
        images = []
        images.append(image)
        images.append(tf.image.flip_up_down(image))
        images.append(tf.image.flip_left_right(image))
        images.append(tf.image.random_brightness(image, max_delta = max_delta))
        
        labels = []
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        
        return images, labels
    '''  
    
    def test_data(self, file_path):
        """
        Description
        -----------
        Reads in test data. The format must match with the image_format instance variable where if 
        the class was created to read binary images for the training set, then the training set 
        must also be a binary file.
        
        Parameters
        ----------
        file_path: string
            a string var specifying where the file is located
            i.e. './cifar-100-binary/test.bin'
                 './cifar-100-python/test'
        
        Returns
        -------
        test_image
            returns a numpy array of the test images
        test_label
            returns a numpy vector of the labels corresponding to the test images
        """
        if(self.image_format == 0):
            # Unpickle the file and store the data into a dictionary 
            with open(file_path, 'rb') as fo:
                dict = pickle.load(fo, encoding = 'bytes')
        
            # Extract the images from the dictionary
            test_image = np.array(dict[b'data'].reshape([-1, 3, 32, 32]))
            test_image = test_image.transpose([0, 2, 3, 1])
        
            # Extract the labels from the dictionary
            test_label = np.array(dict[b'fine_labels'])

        elif(self.image_format == 1):
            data = np.fromfile(file_path, dtype = np.uint8)
            label_index = self.label_size - 1 # in the case of cifar-100, there are 2 labels and so we take the fine label which is at index 1
            image_index = label_index + 1 # image pixels start at the index after the fine label hence we add 1
            
            test_image = []
            test_label = []
            
            while(label_index < data.shape[0] or image_index < data.shape[0]):
                test_label.append(data[label_index])
                test_image.append(data[image_index:(image_index + self.image_height * self.image_width * self.image_depth)])
            
                label_index += 2 + self.image_height * self.image_width * self.image_depth
                image_index += 2 + self.image_height * self.image_width * self.image_depth
                
            test_image = np.array(test_image)
            test_label = np.array(test_label)
            
            test_image = np.reshape(test_image, [-1, self.image_depth, self.image_width, self.image_height])
            test_image = test_image.transpose([0, 2, 3, 1])
            
        elif(self.image_format == 2):
            pass
        else:
            raise Exception("Error in test_data func: image_format isn't 0, 1, or 2. Invalid number!")
            
        return test_image, test_label
    
    def next_batch(self):
        """
        Description
        -----------
        Returns a batch of images and their corresponding labels. 
        The pickled file is read as a numpy array but the binary file is read as a tensor since queueing returns a tensor.
        
        Returns
        -------
            batch_image
                if pickle file is read, returns a 4D numpy array (shaped Batch_size, Width, Height, Channels) of images
                if binary file is read, a 4D tensor (shaped Batch_size, Width, Height, Channels) of images is returned
            batch_label
                if pickle file is read, returns a numpy vector of labels
                if binary file is read, a tensor vector of labels are returned
        """
        if(self.image_format == 0):
            batch_image = []
            batch_label = []
            for _ in range(self.batch_size):
                image, label = self.__read_image()
                # images, labels = self.image_preprocessing(image, label)
                batch_image.append(image)
                batch_label.append(label)
            
        else:
            image, label = self.__read_image()
            # images, labels = self.image_preprocessing(image, label)
            
            batch_image, batch_label = tf.train.shuffle_batch([image, label], batch_size = self.batch_size, 
                                                          capacity = self.capacity, min_after_dequeue = self.min_after_dequeue, 
                                                          num_threads = self.num_threads)
            batch_label = tf.reshape(batch_label, [self.batch_size])
            
        return batch_image, batch_label
    