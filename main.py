import argparse
import time
import numpy as np
from mindspore import context, Tensor
from mindspore import dtype as mstype 
from mindspore import nn, ops
from DataSet import GenerateDataSet, GenerateSet
from model import NetG, NetF, LenFac, Loss
from train import train


def ArgParse():
    parser = argparse.ArgumentParser(description="Penalty-Free Neural Network Method")
    parser.add_argument("--problem", type=int, default=1,
                        help="the type of the problem")
    parser.add_argument("--bound", type=float, default=[-1.0,1.0,-1.0,1.0], 
                        help="lower and upper bounds of the domain")
    parser.add_argument("--inset_nx", type=int, default=[60,60],
                        help="size of the inner set")
    parser.add_argument("--bdset_nx", type=int, default=[60,60],
                        help="size of the boundary set")
    parser.add_argument("--teset_nx", type=int, default=[101,101],
                        help="size of the test set")
    parser.add_argument("--g_epochs", type=int, default=5000,
                        help="number of epochs to train neural network g")
    parser.add_argument("--f_epochs", type=int, default=5000,
                        help="number of epochs to train neural network f")
    parser.add_argument("--g_lr", type=float, default=0.01,
                        help="learning rate to train neural network g")
    parser.add_argument("--f_lr", type=float, default=0.01,
                        help="learning rate to train neural network f")
    parser.add_argument("--use_cuda", type=bool, default=False,
                        help="device")
    parser.add_argument("--tests_num", type=int, default=3,
                        help="number of independent tests")
    parser.add_argument("--g_path", type=str, default="./optimal_state/optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="./optimal_state/optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")       
    return args

if __name__ == "__name__" :
    args = ArgParse()
    if args.use_cuda :
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, devide_target="CPU")
    errors = np.array(args.tests_num)
    
    for iter in range(args.tests_num):
        InSet,BdSet,TeSet=GenerateSet(args)
        dsg, dsloss = GenerateDataSet()

        lenfac = LenFac(Tensor(args.bounds,mstype.float32).reshape(2,2),1)
        netg = NetG()
        netf = NetF()
        netloss = Loss(netf)
        optimg = nn.Adam(netg.trainable_params(), learning_rate=args.lr)
        optimf = nn.Adam(netf.trainable_params(), learning_rate=args.lr)

        start_time = time.time(0)
        train(args, netg, netf, netloss, lenfac, optimg, optimf, InSet, BdSet, dsg, dsloss)

        elapsed = time.time() - start_time
        print("train time: %.2f" %(elapsed))

        errors[iter] = eval(netg, netf, lenfac, TeSet)
        print("test_error = %.3e\n" %(errors[iter].item()))
    
    print(errors)
    errors_mean = errors.mean()
    errors_std = errors.std()

    print("test_error_mean = %.3e, test_error_std = %.3e"
            %(errors_mean.item(), errors_std.item()))
