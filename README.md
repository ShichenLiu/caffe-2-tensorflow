# From Caffe to Tensorflow
This is an instruction on how to transfer caffe project to tensorflow. If you are not familiar with tensorflow or don't understand what is going on, please refer to [Tensorflow Section](http://???)

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

If you have to debug, here are something you need to notice:

1. random shuffle: each epoch caffe would random shuffle all data. Modify shuffle option in the prototxt.
2. mirror: at train phase, each image might have 0.5 possibility to be left-right flipped. Modify mirror option in the prototxt.
3. random crop: even though each image have already been resized to 256x256, it will further be cropped to 227x227. One way is to use a monocolor image that is identical to crop, the other is to modify data_transform.cpp.
4. pretrain model: use a pretrain model that can cover all variables.
5. after above, there should be no more randomization in your debugging.

## Convolutional
There are two ways of padding in Tensorflow

1. VALID: *aka.* zero-padding, always starts from top-left, ignoring bottom-most and right-most pixels.
2. SAME: try to even padding at left and right

e.g. Conv1 layer in Alexnet is VALID padding, while others are SAME padding as well as pooling layers.

## Optimizer Strategy

## Validation

## Debugging
