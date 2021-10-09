'''
Define the network of PFNN 
A penalty-free neural network method for solving a class of 
second-order boundary-value problems on complex geometries
'''

from Data import InSet
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Normal

class LenFac(nn.Cell):
    def __init__(self, bounds, mu):
        super(LenFac, self).__init__()
        self.bounds = bounds
        self.hx = self.bounds[0,1] - self.bounds[0,0]
        self.mu = mu
    
    def cal_l(self, x):
        return 1.0 - (1.0-(x-self.bounds[0,0])/self.hx) ** self.mu

    def construct(self, x):
        return self.cal_l(x)


class NetG(nn.Cell):
    def __init__(self):
        super(NetG, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.1), bias_init=Normal(0.1))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.1), bias_init=Normal(0.1))
        self.fc2 = nn.Dense(10, 1, weight_init=Normal(0.1), bias_init=Normal(0.1))
        self.fcx = nn.Dense(2, 10, weight_init=Normal(0.1), bias_init=Normal(0.1))
    
    def network_without_label(self, x):
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h) + h)
        return self.fc2(x)

    def network_with_label(self, x, label):
        x = self.network_without_label(x)
        return ((x - label)**2).mean()

    def construct(self, x, label=None):
        if label is None:
            return self.network_without_label(x)
        else:
            return self.network_with_label(x, label)


class NetF(nn.Cell):
    def __init__(self):
        super(NetF, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.1), bias_init = Normal(0.1))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.1), bias_init = Normal(0.1))
        self.fc2 = nn.Dense(10, 10, weight_init=Normal(0.1), bias_init = Normal(0.1))
        self.fc3 = nn.Dense(10, 10, weight_init=Normal(0.1), bias_init = Normal(0.1))
        self.fc4 = nn.Dense(10, 1, weight_init=Normal(0,1), bias_init = Normal(0.1))
        self.fcx = nn.Dense(2, 10, weight_init=Normal(10,1), bias_init = Normal(0.1))

    def construct(self, x):
        h = self.fx0(x)
        x = self.fc1(h+self.fcx)
        h = self.fc2(x)
        x = self.fc3(h)
        return self.fc4(x)


class Loss(nn.Cell):
    '''
    '''
    def __init__(self, NetF):
        super(Loss, self).__init__()
        #mark matmul
        self.matmul = nn.MatMul()
        self.grad = ops.composite.GradOperation()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.net = NetF

    def get_variable(self, InSet_g, InSet_l, InSet_gx, InSet_lx, InSet_a, InSet_size,
        InSet_dim, InSet_area, InSet_c, BdSet_nlength, BdSet_nr, BdSet_nl, BdSet_ng):
        self.InSet_g = Parameter(Tensor(InSet_g, mstype.float32), name="InSet_g", requires_grad=False)
        self.InSet_l = Parameter(Tensor(InSet_l, mstype.float32), name="InSet_l", requires_grad=False)
        self.InSet_gx = Parameter(Tensor(InSet_gx, mstype.float32), name="InSet_gx", requires_grad=False)
        self.InSet_lx = Parameter(Tensor(InSet_lx, mstype.float32), name="InSet_lx", requires_grad=False)
        self.InSet_a = Parameter(Tensor(InSet_a, mstype.float32), name="InSet_a", requires_grad=False)
        self.InSet_size = Parameter(Tensor(InSet_size, mstype.float32), name="InSet_size", requires_grad=False)
        self.InSet_dim = Parameter(Tensor(InSet_dim, mstype.float32), name="InSet_dim", requires_grad=False)
        self.InSet_area = Parameter(Tensor(InSet_area, mstype.float32), name="InSet_area", requires_grad=False)
        self.InSet_c = Parameter(Tensor(InSet_c, mstype.float32), name="InSet_c", requires_grad=False)
        self.BdSet_nlength = Parameter(Tensor(BdSet_nlength, mstype.float32), name="BdSet_nlength", requires_grad=False)
        self.BdSet_nr = Parameter(Tensor(BdSet_nr, mstype.float32), name="BdSet_nr", requires_grad=False)
        self.BdSet_nl = Parameter(Tensor(BdSet_nl, mstype.float32), name="BdSet_nl", requires_grad=False)
        self.BdSet_ng = Parameter(Tensor(BdSet_ng, mstype.float32), name="BdSet_ng", requires_grad=False)

    def construct(self, InSet_x, BdSet_x):
        InSet_f = self.net(InSet_x)
        InSet_fx = self.grad(self.net)(InSet_x)
        InSet_u = self.InSet_g + self.InSet_l * InSet_f
        InSet_ux = self.InSet_gx + self.InSet_lx * InSet_f + self.InSet_l * InSet_fx
        InSet_aux = self.in_set_a.matmul(InSet_ux.reshape(self.InSet_size, self.InSet_dim, 1))
        InSet_aux = InSet_aux.reshape(self.InSet_size, self.InSet_dim)
        BdSet_nu = self.BdSet_ng + self.BdSet_nl * self.net(BdSet_x)

        return 0.5 * self.InSet.area * self.sum(self.mean((InSet_aux * InSet_ux),0)) + \
            self.InSet_area * self.mean(self.InSet_c * InSet_u) - \
            self.BdSet_nlength * self.mean(self.BdSet_nr * BdSet_nu)
        

