##################################################################################
# 2017 01.16 Created by Shichen Liu                                              #
# Alexnet in tensorflow                                                          #
##################################################################################

class Alexnet(object):
    def __init__(self, net_path):
        '''
        Args:
            net_data: pretrain model (float32)
        '''
        self._net_data = np.load(net_path).item()
        self._initialized = False
        self._lr_mult = None
        self._out_dim = 4096

    def extract(self, img, train_phase=True):
        '''
        alexnet structure
        Args:
            img: [batch_size, w, h, c] 4-D tensor
        Return:
            fc7l: [batch_size, self._output_dim] tensor
        '''
        lr_mult = dict()
        net_data = self._net_data
        self._initialized = True
        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv1'][0])
            conv = tf.nn.conv2d(img, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=net_data['conv1'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### LRN1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.conv1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)
        ### Pool1
        self.pool1 = tf.nn.max_pool(self.lrn1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool1')
        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv2'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.pool1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.get_variable('biases', initializer=net_data['conv2'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### LRN2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.conv2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)
        ### Pool2
        self.pool2 = tf.nn.max_pool(self.lrn2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')
        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv3'][0])
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=net_data['conv3'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv4'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.get_variable('biases', initializer=net_data['conv4'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights', initializer=net_data['conv5'][0])
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.get_variable('biases', initializer=net_data['conv5'][1])
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')
        ### FC6
        ### Output 4096
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', initializer=net_data['fc6'][0])
            fc6b = tf.get_variable('biases', initializer=net_data['fc6'][1])
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            if train_phase:
                self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            else:
                self.fc6 = tf.nn.relu(fc6l)
            lr_mult[fc6w] = 1
            lr_mult[fc6b] = 2
            
        ### FC7
        ### Output 4096
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', initializer=net_data['fc7'][0])
            fc7b = tf.get_variable('biases', initializer=net_data['fc7'][1])
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            if train_phase:
                self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            else:
                self.fc7 = tf.nn.relu(fc7l)
            lr_mult[fc7w] = 1
            lr_mult[fc7b] = 2
        self._lr_mult = lr_mult
        return self.fc7

    @property
    def lr_mult(self):
        assert self._initialized == True, "Alexnet not initialized"
        return self._lr_mult

    @property
    def output_dim(self):
        return self._out_dim
