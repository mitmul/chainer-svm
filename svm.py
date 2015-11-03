from hinge import hinge

import chainer
import chainer.functions as F
from chainer import cuda, Variable


class SVM(chainer.FunctionSet):

    def __init__(self, c, penalty='L1'):
        super(SVM, self).__init__(
            fc=F.Linear(2, 2),
        )
        self.c = c
        self.penalty = penalty

    def forward(self, x, t, train=True):
        xp = cuda.get_array_module(*x)

        x = Variable(x, volatile=not train)
        t = Variable(t, volatile=not train)
        h = self.fc(x)
        loss = hinge(h, t, self.penalty)

        if self.penalty == 'l1':
            loss += self.c * F.sum(Variable(abs(self.fc.W),
                                            volatile=not train))

        elif self.penalty == 'l2':
            n = Variable(self.fc.W.dot(self.fc.W.T), volatile=not train)
            loss += self.c * F.reshape(n, ())

        return loss
