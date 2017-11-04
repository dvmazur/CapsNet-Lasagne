import theano.tensor as T
import theano

from lasagne.layers import Layer, InputLayer
from lasagne.init import GlorotUniform, Constant

class LengthLayer(Layer):

    def __init__(self, incoming, **kwargs):
        super(LengthLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return T.sqrt(T.sum(T.sqr(input), -1))

    def get_output_shape_for(self, input_shape):
        return tuple(self.input_shape[:-1])

def squash(input):
    """
    Basically computes the "length" of the vector outputed by a capsule
    :param input: 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1]
    :return: 5-D tensor with same shape as input, but squashed in dims 4 and 5
    """

    vec_squashed_norm = T.sum(T.square(input), -2, keepdims=True)
    scalar_factor = vec_squashed_norm / (1 + vec_squashed_norm) / T.sqrt(vec_squashed_norm)

    return scalar_factor * input

def softmax(c, dim):
    return T.exp(c) / T.sum(c, axis=dim)


class CapsLayer(Layer):
    """
    Capsule Layer

    :param incoming: Lasagne Layer
    :param num_capsule: int, number of capsules in one layer
    :param dim_vector: int, size of the vector outputed by a single capsule
    """

    def __init__(self, incoming, num_capsule, dim_vector, num_routing=3, W=GlorotUniform(), b=Constant(0), **kwargs):
        super(CapsLayer, self).__init__(incoming, **kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing

        self.input_num_caps = self.input_shape[1]
        self.input_dim_vector = self.input_shape[2]

        self.W = self.add_param(W,
                                (self.input_num_caps, self.num_capsule, self.input_dim_vector, self.dim_vector),
                                name="W")

        self.b = self.add_param(b,
                                (1, self.input_num_caps, self.num_capsule, 1, 1),
                                name="b")


    def get_output_for(self, input, **kwargs):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = T.reshape(input, (-1, self.input_num_caps, 1, 1, self.input_dim_vector))

        # Prepare for matrix multiplication
        inputs_tiled = T.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        inputs_hat, updates = theano.scan(lambda x: T.batched_tensordot(x, self.W, [3, 2]),
                                 sequences=inputs_tiled)

        # the routing algorithm
        for r in range(self.num_routing):
            c = softmax(self.b, -1)
            c_expand = T.reshape(c, [1, self.input_num_caps, self.num_capsule, 1, 1])

            outputs = T.sum(c_expand * inputs_hat, 1, keepdims=True)
            outputs = squash(outputs)

            self.b = self.b + T.sum(inputs_hat * outputs, [0, -2, -1])

        if self.num_routing == 0:
            c = softmax(self.b)
            c_expand = T.reshape(c, [1, self.input_num_caps, self.num_capsule, 1, 1])
            outputs = squash(T.sum(c_expand * inputs_hat, 1, keepdims=True))

        return outputs


    def get_output_shape_for(self, input_shape):
        return tuple([self.input_shape[0], self.num_capsule, self.dim_vector])
