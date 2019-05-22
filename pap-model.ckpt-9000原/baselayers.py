import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

DEFAULT_PADDING = 'SAME'
def layer(op):
    """
    Decorator for composable network layers.
    """

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, num_classes=6, input_size=[320, 320]):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup(is_training, num_classes, input_size)

    def setup(self, is_training):

        """
        Construct the network.
        """

        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):

        """
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """

        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):

        """
        Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """

        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):

        """
        Returns the current network output.
        """

        return self.terminals[-1]

    def get_unique_name(self, prefix):

        """
        Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """

        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, initializer=None):

        """
        Creates a new TensorFlow variable.
        """

        if initializer is None:
            return tf.get_variable(name, shape, trainable=self.trainable)
        else:
            return tf.get_variable(name, shape, trainable=self.trainable, initializer=initializer)

    def validate_padding(self, padding):

        """
        Verifies that the padding is one of the supported ones.
        """

        assert padding in ('SAME', 'VALID')

    def get_bilinear_filter(self, filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        # bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
        #                       shape=weights.shape)
        bilinear_weights = self.make_var('weights_bilinear', shape=weights.shape, initializer=init)
        return bilinear_weights

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape().as_list()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def upsample(self, bottom, n_channels_in, n_channels_out, upscale_factor, name, relu=True, biased=True):

        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name) as scope:
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            h = in_shape[1] * stride
            w = in_shape[2] * stride
            new_shape = [in_shape[0], h, w, n_channels_out]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels_in, n_channels_out]

            weights = self.get_bilinear_filter(filter_shape, upscale_factor)
            output = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')
        if biased:
            biases = self.make_var('biases', [n_channels_out])
            output = tf.nn.bias_add(output, biases)
        if relu:
            # ReLU non-linearity
            output = tf.nn.relu(output)

        return tf.reshape(output, output_shape, name=scope.name)

    @layer
    def deconv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        output_shape = [input.get_shape().as_list()[0], input.get_shape().as_list()[1] * s_h,
                        input.get_shape().as_list()[2] * s_w, c_o]
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_o, c_i / group])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def upsize(self,
             input,
             old_width,
             old_height,
             scale,
             name):
        new_height = int(round(old_height * scale))
        new_width = int(round(old_width * scale))
        resized = tf.image.resize_images(input, [new_width, new_height])
        resized = tf.identity(resized, name=name)
        return resized

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name, weight=None):
        if weight is None:
            return tf.add_n(inputs, name=name)
        else:
            assert len(weight) == len(inputs)
            result = weight[0] * inputs[0]
            for index in range(1, len(inputs)):
                result = tf.add(result, weight[index] * inputs[index])
            return result

    @layer
    def free_add(self, inputs, name, identity=True):
        with tf.variable_scope(name) as scope:
            if identity:
                return tf.add_n(inputs, name=name)
            else:
                weights = []
                for index in range(0, len(inputs)):
                    weights.append(self.make_var('fc_weights_' + str(index), shape=[1],
                                                 initializer=tf.constant_initializer(1.0/len(inputs))))
                result = weights[0] * inputs[0]
                for index in range(1, len(inputs)):
                    result = tf.add(result, weights[index] * inputs[index])
                return result

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = slim.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def groupnorm(self, input, G, name, eps=1e-5):
        with tf.variable_scope(name) as scope:
            N, H, W, C = input.shape
            output1 = tf.reshape(input, [tf.to_int32(N), tf.to_int32(H), tf.to_int32(W), G, tf.to_int32(C / G)])
            mean, var = tf.nn.moments(output1, [1, 2, 4], keep_dims=True)
            output = (output1 - mean) / tf.sqrt(var + eps)
            output = tf.reshape(output, tf.shape(input))
            gamma = self.make_var('gamma', [1, 1, 1, input.shape[3]])
            beta = self.make_var('beta', [1, 1, 1, input.shape[3]])
            return output * gamma + beta

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def sep_conv(self, input, k_h, k_w, channels, s, name):
        with tf.variable_scope(name) as scope:
#            output = tf.nn.depthwise_conv2d(input, [k_h, k_w, channels, 1], strides=s, padding='SAME', name=name)
            output = slim.separable_conv2d(input, channels, [k_h, k_w],depth_multiplier=1, stride=s, rate=1, scope=scope)
            return output
