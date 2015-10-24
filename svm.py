from hinge import hinge

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import Variable


class SVM(chainer.FunctionSet):

    def __init__(self, c, penalty='l1'):
        super(SVM, self).__init__(
            fc=F.Linear(2, 1),
        )
        self.c = c
        self.penalty = penalty

    def forward(self, x, t, train=True):
        xp = cuda.get_array_module(*x)
        num = x.shape[0]
        x = Variable(x, volatile=not train)
        t = Variable(t, volatile=not train)

        h = self.fc(x)
        h = F.reshape(h, (num,))
        loss = hinge(h, t)

        if self.penalty == 'l1':
            loss += self.c * F.sum(Variable(self.fc.W, volatile=not train))
            
        elif self.penalty == 'l2':
            W = Variable(self.fc.W, volatile=not train)
            W_T = F.transpose(W, (1, 0))
            loss += self.c * F.reshape(F.matmul(W, W_T), ())

        return loss
