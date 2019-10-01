"""
Bundle-based Hybrid Parallelism test code.
This code is used to test and describe its action.
We didn't do code optimization especially in multiprocessing part.
Some communication hiding techniques are required to optimize its performance.

    - last update: 2019.09.30
    - E.Jubilee Yang
"""
from HPBundle import Hybrid_Bundle, Worker
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
parser.add_argument('--IP',default='127.0.0.1',type=str,
                    help='URL used to set up distributed training')
parser.add_argument('--portNum',default='8888',type=str,
                    help='Port number for the distributed training')
parser.add_argument('--model',default='resnet101_trial1',
                    type=str)
parser.add_argument('--world-size',type=int,default=0,
                    help='Degree of inter processing (number of node)')
parser.add_argument('--num-hp',type=int,default=1,
                    help='Degree of inter processing (number of node)')
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

    if args.DP_ONLY:
        for i in range(num_hp):
            workers.append(Worker(batch_size=args.batch_size,
                                  num_total_worker=num_hp,
                                  args=args))
        # Get sync Q
        COLLECTIVE_Q, \
        DISTRIBUTE_Q = workers[0].get_sync_channel()

        # Set sync Q
        for i in range(1,num_hp):
            workers[i].set_sync_channel(COLLECTIVE_Q[i-1],
                                        DISTRIBUTE_Q[i-1])

    else:
        for i in range(num_hp):
            workers.append(Hybrid_Bundle(batch_size=args.batch_size,
                                     num_hp=num_hp,
                                     args=args))

        # Get sync Q
        FRONT_COLLECTIVE_Q, \
        FRONT_DISTRIBUTE_Q, \
        REAR_COLLECTIVE_Q, \
        REAR_DISTRIBUTE_Q = workers[0].get_sync_channel()

        # Set sync Q
        for i in range(1,num_hp):
            workers[i].set_sync_channel(FRONT_COLLECTIVE_Q[i-1],
                                        FRONT_DISTRIBUTE_Q[i-1],
                                        REAR_COLLECTIVE_Q[i-1],
                                        REAR_DISTRIBUTE_Q[i-1])
    for i in range(num_hp):
        # Run all Bundle Processes
        p = mp.Process(target=workers[i].run)
        p.start()
        process.append(p)

    for p in process:
        p.join()


if __name__ == '__main__':

    args = parser.parse_args()
    main(args)