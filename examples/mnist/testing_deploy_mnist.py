import os
import numpy as np
import sys
import cv2

# Make sure that caffe is on the python path:
#CAFFE_ROOT = '/home/caffe/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH
CAFFE_ROOT = '/media/nileshm/B2A07268A07232C9/Users/niles/Downloads/Uni/PartIV/Code/ubuntuCaffe/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_ROOT + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = CAFFE_ROOT + 'examples/mnist/lenet_deploy.prototxt'
PRETRAINED = CAFFE_ROOT + 'examples/mnist/lenet_iter_10000.caffemodel' 

IMAGE_FILE1 = CAFFE_ROOT + 'examples/mnist/mnistasjpg/testSample/img_1_is2.jpg'
IMAGE_FILE2 = CAFFE_ROOT + 'examples/mnist/mnistasjpg/testSample/img_2_is0.jpg'
IMAGE_FILE3 = CAFFE_ROOT + 'examples/mnist/mnistasjpg/testSample/img_3_is9.jpg'
IMAGE_FILE4 = CAFFE_ROOT + 'examples/mnist/mnistasjpg/testSample/test_3.png'


assert os.path.exists(MODEL_FILE), "MODELFILE not found" 

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
transformer = caffe.io.Transformer({'data': (1, 1, 28, 28)})
transformer.set_transpose('data', (2, 0, 1))    
transformer.set_raw_scale('data', 1/255.)

assert os.path.exists(IMAGE_FILE4), "image %s not found" % args.image
image = cv2.imread(IMAGE_FILE4)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = 255 - image
image.resize((28, 28, 1))
cv2.imshow('image', image)

net.blobs['data'].reshape(1, 1, 28, 28)
net.blobs['data'].data[...] = transformer.preprocess('data', image)    

net.forward()
scores = net.blobs['ip2'].data
print scores
print scores.argmax()
cv2.waitKey(0)