import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Hinge(function.Function):

    """Hinge loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        self.xt = xp.array(x * t, dtype=x.dtype)
        loss = xp.array(xp.sum(xp.maximum(0, 1 - self.xt)), dtype=x.dtype)
        loss /= x.shape[0]

        return loss,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = numpy.zeros_like(t, dtype=numpy.float32)
        wrong_pos = numpy.where(self.xt < 1)
        gx[wrong_pos] = gloss * -t[wrong_pos] / t.shape[0]

        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = cuda.elementwise(
            'T xt, S t, raw T gloss, T inv_cnt', 'T gx',
            '''
            if (xt < 1) gx = -gloss[0] * inv_cnt * t;
            else gx = 0
            ''', 'hinge_bwd')(self.xt, t, gloss, 1.0 / t.shape[0])

        return gx, None


def hinge(x, t, use_cudnn=True):
    return Hinge(use_cudnn)(x, t)
