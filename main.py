"""
Bundle-based Hybrid Parallelism test code.
This code is used to test and describe its action.
We didn't do code optimization especially in multiprocessing part.
Some communication hiding techniques are required to optimize its performance.

    - last update: 2019.09.30
    - E.Jubilee Yang
"""
from HPBundle   import HP_BUNDLE
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description='Bundle-HP')

# Parameter Setting
parser.add_argument('--batch-size',type=int,default=32,
                    help='Input batch size for this node')
parser.add_argument('--rank',type=int,default=0,
                    help='Rank of a node')
parser.add_argument('--lr',type=float,default=0.01,metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum',default=0.9,type=float,metavar='M',
                    help='momentum')
parser.add_argument('--itr',type=int,default=50,
                    help='test iteration')
parser.add_argument('--node-rank',type=int,default=0,
                    help='rank of this node')
parser.add_argument('--IP',default='127.0.0.1',type=str,
                    help='URL used to set up distributed training')
parser.add_argument('--portNum',default='8888',type=str,
                    help='Port number for the distributed training')
parser.add_argument('--model',default='resnet101_trial1',
                    type=str)
parser.add_argument('--world-size',type=int,default=0,
                    help='Degree of inter processing (number of node)')
parser.add_argument('--num-hp',type=int,default=1,
                    help='Degree of inter processing (number of hp)')
parser.add_argument('--weight-decay','--wd',default=1e-4,type=float,metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--DP-ONLY',dest='DP_ONLY',action='store_true',
                    help='boolean flag for data parallelism only ')
parser.add_argument('data',metavar='DIR',
                    help='path to dataset')

def main(args):

    workers = []
    process = []
    num_hp = args.num_hp

    mp.set_start_method('spawn')

    hybrid_bundle = HP_BUNDLE(shape=[1,1],
                              num_bundles=args.num_hp,
                              num_nodes=args.world_size,
                              rank=0,
                              args=args)
    hybrid_bundle.run()

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)