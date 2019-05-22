# encoding=utf8
import sys  

# reload(sys)  
# sys.setdefaultencoding('gbk')

import os
#init tf
from datetime import datetime
import os
import time
import cv2
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

WIDTH = 224
HIGHT = 224
NUM_CLASSES = 3

RESTORE_FROM = 'model.ckpt-9000'

"""Create the model and start the evaluation process."""

with tf.name_scope("create_inputs"):
    image = tf.placeholder(tf.float32, [HIGHT, WIDTH, 3])
    image_mean = image - IMG_MEAN
    image_batch = tf.expand_dims(image_mean, dim=0)
    # Add one batch dimension.

# Create network.
net = DeepLabResNetModel({'data': image_batch},
                            is_training=False, num_classes=NUM_CLASSES, input_size=[HIGHT, WIDTH])

# Which variables to load.
restore_var = tf.global_variables()

# Predictions.
raw_output = tf.nn.softmax(net.layers['upscore'])
raw_output = tf.reshape(raw_output, [-1, NUM_CLASSES])

# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)
sess.run(tf.local_variables_initializer())

# Load weights.
loader = tf.train.Saver(var_list=restore_var)

loader.restore(sess, RESTORE_FROM)
print(("Restored model parameters from {}".format(RESTORE_FROM)))

start_time = time.time()

img_fn = 'FX00000.JPG'
image_pred0 = cv2.imread(img_fn, cv2.IMREAD_COLOR)
image_pred = cv2.resize(image_pred0, (WIDTH, HIGHT))

prediction_result = sess.run(raw_output, feed_dict={image: image_pred})
print(raw_output)
print(prediction_result)
if np.argmax(prediction_result) == 0:
    print("Nugent: 0-3 score")
elif np.argmax(prediction_result) == 1:
    print("Nugent: 4-6 score")
elif np.argmax(prediction_result) == 2:
    print("Nugent: 7-10 score" + "  BV(+)")
else:
    print("No Scores")

