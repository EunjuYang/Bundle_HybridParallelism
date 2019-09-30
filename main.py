from HPBundle import Hybrid_Bundle
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
parser.add_argument('--dp-only',default=False,type=bool,
                    help='boolean flag for data parallelism only ')
parser.add_argument('data',metavar='DIR',
                    help='path to dataset')

def main(args):

    hybrids = []
    process = []
    num_hp = args.num_hp

    mp.set_start_method('spawn')

    for i in range(num_hp):
        hybrids.append(Hybrid_Bundle(batch_size=32,
                                     num_hp=num_hp,
                                     args=args))


    FRONT_COLLECTIVE_Q, \
    FRONT_DISTRIBUTE_Q, \
    REAR_COLLECTIVE_Q, \
    REAR_DISTRIBUTE_Q = hybrids[0].get_sync_channel()

    for i in range(num_hp):

        # Setting for Intra-DP
        if i is not 0:
            hybrids[i].set_sync_channel(FRONT_COLLECTIVE_Q[i-1],
                                        FRONT_DISTRIBUTE_Q[i-1],
                                        REAR_COLLECTIVE_Q[i-1],
                                        REAR_DISTRIBUTE_Q[i-1])
        # Run all Bundle Processes
        p = mp.Process(target=hybrids[i].run,
                       args=())
        p.start()
        process.append(p)

    for p in process:
        p.join()


if __name__ == '__main__':

    args = parser.parse_args()
    main(args)