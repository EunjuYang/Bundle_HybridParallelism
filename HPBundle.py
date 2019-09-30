"""
This code contains code for hybrid parallelism (HP) parameter server.
One Hybrid_PS contains two PSs - parameter server for front and rear.


"""
from termcolor import colored
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import model
import torch.multiprocessing as mp
import numpy as np
import time, torch, os,random, threading


class Hybrid_Bundle():

    # << Caution >>
    # In spawned process, class variable will be initialized.
    # Do not refer class variable in new process but use object variable with self.
    NUM_HP = 0


    def __init__(self, batch_size, num_hp, args=None):

        if args.DP_ONLY:
            self.MP_MODE = 1
        else:
            # Mode of Model Parallelism
            # Default value two denotes there exist two part of model in this HP
            MP_MODE = 2
            self.MP_MODE = 2

        self.DP_ONLY = args.DP_ONLY

        # rank of hybrid parameter server
        self.bundle_local_rank = self.NUM_HP

        # batch size of this hybrid pair
        self.bs = batch_size

        # total number of hybrid pair in this node
        self.num_hp = num_hp

        # inter HP node list (rank list)
        self.inter_sync_front   = [self.MP_MODE * i for i in range(args.world_size)]
        self.inter_sync_rear    = [self.MP_MODE * i + 1 for i in range(args.world_size)]
        self.INTER_DP = True if args.world_size > 1 else False


        # Test num_hp range check
        if torch.cuda.device_count() < self.num_hp * self.MP_MODE:
            num_gpus = torch.cuda.device_count()
            print(colored(' <ERROR!>', "red"),
                  colored('Number of GPUs (%d) are insufficient to support hybrid_bundle with degree %d \n'% (num_gpus, self.num_hp),'yellow'),
                  colored('<ERROR!>', "red"),
                  colored('At least %d GPUs are required' % (num_hp * self.MP_MODE), "yellow"))
            exit()

        # master bool var
        self.MASTER = True if self.bundle_local_rank == 0 else False

        # args
        self.args = args

        # update the number of hybrid parameter server running in this node
        Hybrid_Bundle.NUM_HP += 1

        # if MASTER of hybrid parallelism
        if self.MASTER:

            self.FRONT_COLLECTIVE_Q = []
            self.FRONT_DISTRIBUTE_Q = []
            self.REAR_COLLECTIVE_Q = []
            self.REAR_DISTRIBUTE_Q = []

            # MASTER will collect & distribute for synchronization
            for _ in range(self.num_hp-1):
                self.FRONT_COLLECTIVE_Q.append(mp.Queue())
                self.FRONT_DISTRIBUTE_Q.append(mp.Queue())
                self.REAR_COLLECTIVE_Q.append(mp.Queue())
                self.REAR_DISTRIBUTE_Q.append(mp.Queue())

        # If Hybrid Parallelism (MP)
        if not self.DP_ONLY:
            self.MP_FORWARD_Q = mp.Queue()
            self.MP_BACKWARD_Q = mp.Queue()



    def get_sync_channel(self):

        if not self.MASTER:
            return -1, -1, -1, -1
        else:
            return self.FRONT_COLLECTIVE_Q, self.FRONT_DISTRIBUTE_Q, self.REAR_COLLECTIVE_Q, self.REAR_DISTRIBUTE_Q


    def set_sync_channel(self, front_collective, front_distribute, rear_collective, rear_distribute):

        if self.MASTER:
            return
        self.FRONT_COLLECTIVE_Q = front_collective
        self.FRONT_DISTRIBUTE_Q = front_distribute
        self.REAR_COLLECTIVE_Q = rear_collective
        self.REAR_DISTRIBUTE_Q = rear_distribute

    def run(self):

        processes  = []
        print("=====> (Bundle %d) start"%self.bundle_local_rank)

        if self.DP_ONLY:
            p = mp.Process(target=self.worker)
            p.start()
            processes.append(p)

        else:
            # Front Server
            p = mp.Process(target=self.front)
            p.start()
            processes.append(p)

            # Rear Server
            p = mp.Process(target=self.rear)
            p.start()
            processes.append(p)

        # Join two servers
        for p in processes:
            p.join()


    def worker(self):

        # Set GPU rank (index)
        gpu_rank    = self.bundle_local_rank
        model_name  = self.args.model
        Net         = getattr(model, model_name)

        # Setting INTER DP
        if self.MASTER:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank) * self.MP_MODE,
                                    world_size = self.args.world_size * self.MP_MODE)
            self.front_sync_group = dist.new_group(self.inter_sync_front)

        print("=====> (DP ONLY Bundle %d); worker start with GPU %d "% (self.bundle_local_rank, gpu_rank))

        # Data Preparation
        train_dir   = os.path.join(self.args.data, 'train')
        val_dir     = os.path.join(self.args.data, 'val')
        normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229,0.224,0.225])

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = self.bs,
            shuffle     = True,
            num_workers = self.num_hp,
            pin_memory  = True,
        )


        # For model parallelism; data shuffle matching
        SEED = 777 + self.bundle_local_rank
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        # Setting for forward
        torch.cuda.set_device(gpu_rank)

        # Declare network model and allocate its memory on GPU
        net = Net().cuda(gpu_rank)

        # Define optimizer to be used in training
        optimizer   = optim.SGD(net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss().cuda(gpu_rank)

        for batch_idx, (data, target) in enumerate(train_loader):

            print("-----(BUNDLE %d) worker itr %d-----" % (self.bundle_local_rank, batch_idx))

            # load on gpu
            data = data.cuda(gpu_rank)
            target = target.cuda(gpu_rank)

            # feed forward
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            torch.cuda.synchronize(gpu_rank)

            # syncrhonize with intra-nodes
            if batch_idx == 0 and self.MASTER:
                self.front_ps = []
                for name, layer in net.named_parameters():
                    self.front_ps.append(layer.grad.data.cpu().detach())

            self.front_synchronization(net)

            # apply the update
            optimizer.step()

            del data, target

            if batch_idx == self.args.itr:
                return


    def front(self):

        # Set GPU rank (index)
        gpu_rank    = self.bundle_local_rank * self.MP_MODE

        # Set network model for front
        model_name  = self.args.model + "_front"
        Net         = getattr(model, model_name)

        # Setting INTER HP
        if self.MASTER and self.INTER_DP:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank) * self.MP_MODE,
                                    world_size = self.args.world_size * self.MP_MODE)
            self.front_sync_group = dist.new_group(self.inter_sync_front)
            self.rear_sync_group  = dist.new_group(self.inter_sync_rear)


        print("=====> (Bundle %d); front start with GPU %d "% (self.bundle_local_rank, gpu_rank))

        # Data Preparation
        train_dir   = os.path.join(self.args.data, 'train')
        val_dir     = os.path.join(self.args.data, 'val')
        normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229,0.224,0.225])

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = self.bs,
            shuffle     = True,
            num_workers = self.num_hp,
            pin_memory  = True,
        )


        # For model parallelism; data shuffle matching
        SEED = 777 + self.bundle_local_rank
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        # Setting for forward
        torch.cuda.set_device(gpu_rank)

        # Declare network model and allocate its memory on GPU
        net = Net().cuda(gpu_rank)

        # Define optimizer to be used in training
        optimizer   = optim.SGD(net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)

        for batch_idx, (data, target) in enumerate(train_loader):

            print("-----(BUNDLE %d) front itr %d-----" % (self.bundle_local_rank, batch_idx))

            # load on gpu
            data = data.cuda(gpu_rank)

            # feed forward
            output = net(data)

            # send feed forward output to rear
            self.MP_FORWARD_Q.put(output.cpu().detach())

            # receive backpropagation input from rear
            backward_input_tmp = self.MP_BACKWARD_Q.get()
            backward_input = backward_input_tmp.cuda(gpu_rank)

            # backpropagation
            output.backward(backward_input)

            # synchronization
            if batch_idx == 0 and self.MASTER:
                self.front_ps = []
                for name, layer in net.named_parameters():
                    self.front_ps.append(layer.grad.data.cpu().detach())

            self.front_synchronization(net)
            optimizer.step()

            # Memory garbage collection
            del backward_input_tmp

            if batch_idx == self.args.itr:
                return


        return

    def rear(self):

        # Set GPU rank (index)
        gpu_rank    = self.bundle_local_rank * 2 + 1

        # Set network model for front
        model_name  = self.args.model + "_rear"
        Net         = getattr(model, model_name)

        # Setting INTER HP
        if self.MASTER and self.INTER_DP:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank) * self.MP_MODE + 1,
                                    world_size = self.args.world_size * self.MP_MODE)
            self.front_sync_group = dist.new_group(self.inter_sync_front)
            self.rear_sync_group  = dist.new_group(self.inter_sync_rear)

        print("=====> (Bundle %d); rear start with GPU %d "% (self.bundle_local_rank, gpu_rank))

        # Data Preparation
        train_dir   = os.path.join(self.args.data, 'train')
        val_dir     = os.path.join(self.args.data, 'val')
        normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229,0.224,0.225])

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = self.bs,
            shuffle     = True,
            num_workers = self.num_hp,
            pin_memory  = True,
        )

        # For model parallelism; data shuffle matching
        SEED = 777 + self.bundle_local_rank
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        # Setting for forward
        torch.cuda.set_device(gpu_rank)

        # Declare network model and allocate its memory on GPU
        net = Net().cuda(gpu_rank)

        # Define optimizer to be used in training
        optimizer   = optim.SGD(net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss().cuda(gpu_rank)

        for batch_idx, (data, target) in enumerate(train_loader):

            print("-----(BUNDLE %d) rear itr %d-----" % (self.bundle_local_rank, batch_idx))

            # GET forward input (MP)
            forward_input_tmp = self.MP_FORWARD_Q.get()
            forward_input_tensor = forward_input_tmp.clone().cuda(gpu_rank)
            forward_input = Variable(forward_input_tensor, requires_grad=True).cuda(gpu_rank)
            del forward_input_tmp

            # FEED FORWARD
            forward_output = net(forward_input)
            target = target.cuda(gpu_rank)
            torch.cuda.synchronize(gpu_rank)

            loss = criterion(forward_output, target)
            loss.backward()
            torch.cuda.synchronize(gpu_rank)

            # SEND backward output (MP)
            self.MP_BACKWARD_Q.put(forward_input.grad.data)

            # synchronization
            if batch_idx == 0 and self.MASTER:
                self.rear_ps = []
                for name, layer in net.named_parameters():
                    self.rear_ps.append(layer.grad.data.cpu().detach())
            self.rear_synchronization(net)

            # update
            optimizer.step()

            # memory garbage collection
            del forward_input, forward_output, target

            if batch_idx == self.args.itr:
                return


    def rear_synchronization(self, model):

        if self.MASTER:
            threads = []

            # COLLECT
            for i in range(self.num_hp):
                # MASTER
                if i == 0:
                    t = threading.Thread(target=self.sync_local_upload,
                                         args=(model, self.rear_ps))
                # For other workers
                else:
                    t = threading.Thread(target=self.sync_collect,
                                         args=(self.REAR_COLLECTIVE_Q[i-1], self.rear_ps))
                t.start()
                threads.append(t)

            # INTER SYNC
            if self.INTER_DP:
                for grad in self.rear_ps:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.rear_sync_group)

            # DISTRIBUTE
            for i in range(self.num_hp):
                # MASTER
                if i == 0:
                    t = threading.Thread(target=self.sync_local_download,
                                         args=(model, self.rear_ps))
                else:
                    t = threading.Thread(target=self.sync_distribute,
                                         args=(self.REAR_DISTRIBUTE_Q[i-1], self.rear_ps))
                t.start()
                threads.append(t)

        else:
            self.sync_upload(self.REAR_COLLECTIVE_Q,model)
            self.sync_download(self.REAR_DISTRIBUTE_Q, model)

    def front_synchronization(self, model):

        if self.MASTER:
            threads = []

            # COLLECT
            for i in range(self.num_hp):
                # MASTER
                if i == 0:
                    t = threading.Thread(target=self.sync_local_upload,
                                         args=(model, self.front_ps))
                # For other workers
                else:
                    t = threading.Thread(target=self.sync_collect,
                                         args=(self.FRONT_COLLECTIVE_Q[i-1], self.front_ps))
                t.start()
                threads.append(t)

            for t  in threads:
                t.join()


            # INTER SYNC
            if self.INTER_DP:
                for grad in self.front_ps:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.front_sync_group)

            threads = []

            # DISTRIBUTE
            for i in range(self.num_hp):
                # MASTER
                if i == 0:
                    t = threading.Thread(target=self.sync_local_download,
                                         args=(model, self.front_ps))
                else:
                    t = threading.Thread(target=self.sync_distribute,
                                         args=(self.FRONT_DISTRIBUTE_Q[i-1], self.front_ps))
                t.start()
                threads.append(t)

            for t  in threads:
                t.join()

        else:
            self.sync_upload(self.FRONT_COLLECTIVE_Q,model)
            self.sync_download(self.FRONT_DISTRIBUTE_Q, model)


    def sync_collect(self, queue, ps):

        for param in ps:
            tmp = queue.get()
            param += tmp
            del tmp

    def sync_distribute(self, queue, ps):

        for param in ps:
            queue.put(param)

    def sync_local_download(self, model, ps):
        """
        FOR MASTER
        :param model:
        :return:
        """
        for idx, layer in enumerate(model.parameters()):
            layer.grad.data.copy_(ps[idx])

    def sync_local_upload(self, model, ps):
        """
        FOR MASTER
        :param model:
        :return:
        """
        for idx, layer in enumerate(model.parameters()):
            ps[idx] += layer.grad.cpu()


    def sync_upload(self, queue, model):
        """
        For NON-MASTER
        :param queue:
        :param model:
        :return:
        """
        for layer in model.parameters():
            queue.put(layer.grad.cpu().detach())

    def sync_download(self, queue, model):
        """
        For NON-MASTER
        :param queue:
        :param model:
        :return:
        """

        for layer in model.parameters():
            tmp = queue.get()
            layer.grad.data.copy_(tmp.data)
            del tmp



