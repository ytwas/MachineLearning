# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from baselayers import Network
import tensorflow as tf


class DeepLabResNetModel(Network):
    def setup(self, is_training, num_classes, input_size):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
         .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1-1'))
        (self.feed('conv1-1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2b'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2b')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2b'))

        (self.feed('res2a_relu',
                   'bn2b_branch2b')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2b_relu')
         .conv(3, 3, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2b'))

        (self.feed('bn3a_branch1',
                    'bn3a_branch2b')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b_branch2b'))

        (self.feed('res3a_relu',
                  'bn3b_branch2b')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b_relu')
         .conv(3, 3, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2b'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2b')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b_branch2a')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b_branch2b'))

        (self.feed('res4a_relu',
                  'bn4b_branch2b')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b_relu')
         .conv(3, 3, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2b'))

        (self.feed('bn5a_branch1',
                  'bn5a_branch2b')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2b'))

        (self.feed('res5a_relu',
                   'bn5b_branch2b')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .avg_pool(7, 7, 1, 1, padding='VALID', name='poolup')        
         .conv(1, 1, num_classes, 1, 1, biased=False, relu=False, name='xupscore'))
       
        (self.feed('xupscore', 'xupscore')
         .free_add(identity=False, name='upscore'))
