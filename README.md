# From Caffe to Tensorflow
This is an instruction on how to transfer caffe project to tensorflow. If you are not familiar with tensorflow or don't understand what is going on, please refer to [Tensorflow Section](http://???)
**Under construction, not finished yet**

## Basic knowledge
Let me clarify some principles at first. If you have an existing project implemented on Caffe, and you want to try something new on Tensorflow, where you hope Tensorflow to behave exactly the same as Caffe, then you can keep reading. Remember, when you meet any problem and have to debug, you should diminish all randomizations which I will explain as below one by one. Notice that some of the scripts are found in stackoverflow. Finally an example will be given to you.

## Load Image
As we know, caffe use **BGR** channels as input. So if you are using *cv2* lib to load image, then don't worry, because this lib also read image as **BGR**, otherwise you should take care. However, there is another problem on loading image, the *mean file*. There are two ways to set mean in caffe

1. mean value: 3 numbers are given which indicates three channels' mean value, you only need to subtract respectively.
2. mean file: a mean file is givin, typically a *binaryproto* file. This is a pixelwise mean file, it is simply an *uint8* [3, width, height] matrix. The best way is to convert this file to numpy matrix and save as *npy* file. To achieve this with below script, *pycaffe* is needed.

```python
import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('data/ilsvrc12/imagenet_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
array = np.array(caffe.io.blobproto_to_array(blob))
out = array[0]
np.save('imagenet_mean.npy', out)
```

Then you can load images with following codes

```python
def preprocess_img(self, img, batch_size, train_phase, oversample=False):
    '''
    pre-process input image:
    Args:
        img: 4-D tensor
        batch_size: Int 
        train_phase: Bool
    Return:
        distorted_img: 4-D tensor
    '''
    reshaped_image = tf.cast(img, tf.float32)
    mean = tf.constant(np.load(self.mean_file), dtype=tf.float32, shape=[1, 256, 256, 3])
    reshaped_image -= mean
    crop_height = IMAGE_SIZE
    crop_width = IMAGE_SIZE
    if train_phase:
        distorted_img = tf.pack([tf.random_crop(tf.image.random_flip_left_right(each_image), [crop_height, crop_width, 3]) for each_image in tf.unpack(reshaped_image)])
    else:
        if oversample:
            distorted_img1 = tf.pack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 0, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img2 = tf.pack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 28, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img3 = tf.pack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 0, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img4 = tf.pack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 28, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img5 = tf.pack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 14, 14, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img6 = tf.pack([tf.image.crop_to_bounding_box(each_image, 0, 0, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img7 = tf.pack([tf.image.crop_to_bounding_box(each_image, 28, 28, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img8 = tf.pack([tf.image.crop_to_bounding_box(each_image, 28, 0, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img9 = tf.pack([tf.image.crop_to_bounding_box(each_image, 0, 28, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
            distorted_img0 = tf.pack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])            
            distorted_img = tf.concat(0, [distorted_img1, distorted_img2, distorted_img3, distorted_img4, distorted_img5, distorted_img6, distorted_img7, distorted_img8, distorted_img9, distorted_img0])
        else:
            distorted_img = tf.pack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unpack(reshaped_image)])
    return distorted_img
```

If you have to debug, here are something you need to notice:

1. random shuffle: each epoch caffe would random shuffle all data. Modify shuffle option in the prototxt.
2. mirror: at train phase, each image might have 0.5 possibility to be left-right flipped. Modify mirror option in the prototxt.
3. random crop: even though each image have already been resized to 256x256, it will further be cropped to 227x227. One way is to use a monocolor image that is identical to crop, the other is to modify data_transform.cpp.
4. pretrain model: use a pretrain model that can cover all variables.
5. drop out:
6. after above, there should be no more randomization in your debugging.

## Convolutional
There are two ways of padding in Tensorflow

1. VALID: *aka.* zero-padding, always starts from top-left, ignoring bottom-most and right-most pixels.
2. SAME: try to even padding at left and right

e.g. Conv1 layer in Alexnet is VALID padding, while others are SAME padding as well as pooling layers.

## Optimizer Strategy

There are only one formula given by tensorflow, however you can make your own optimizer strategy with it.
e.g.
```python
### Step
self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_factor)
### Inv
self.lr = tf.train.exponential_decay(self.learning_rate, -0.75, 1.0, 1.0+0.002*tf.cast(self.global_step, tf.float32))
```

## Train & Validation

You can extract features using Alexnet and then classify like this
```python
source_img = tf.placeholder(tf.float32, 
    [batch_size, 256, 256, 3])
test_img = tf.placeholder(tf.float32, 
    [1, 256, 256, 3])
source_label = tf.placeholder(tf.float32, 
    [batch_size, n_class])
test_label = tf.placeholder(tf.float32, 
    [1, n_class])
global_step = tf.Variable(0, trainable=False)
### Construct CNN
cnn = Alexnet(model_weights)
### Construct train net
source_img = preprocess_img(source_img, batch_size, True)
with tf.variable_scope("cnn"):
    source_feature = cnn.extract(source_img)
lr_mult = cnn.lr_mult
with tf.variable_scope("classifier"):
    source_fc8 = classifier(source_feature)
### Construct test net
log('setup', 'Construct Test Net')
test_img = preprocess_img(test_img, 1, False)
with tf.variable_scope("cnn", reuse=True):
    test_feature = cnn.extract(test_img, train_phase=False)
with tf.variable_scope("classifier", reuse=True):
    test_fc8 = classifier(test_feature)
test_output = tf.reduce_mean(test_fc8, 0)
test_result = tf.equal(tf.argmax(test_output, 0), tf.argmax(test_label, 1))
```
