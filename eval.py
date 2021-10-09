from mindspore import Tensor
import mindspore.dtype import mstype

def eval(netg,netf,lenfac,TeSet):
    x = Tensor(TeSet.x, mstype.float32)
    TeSet_u = (netg(x)+lenfac(Tensor(x[:,0])).reshape(-1,1)* netf(x)).asnumpy()
    error = (((TeSet_u - TeSet.ua)**2).sum() / (TeSet.ua).sum()) ** 0.5
    return error