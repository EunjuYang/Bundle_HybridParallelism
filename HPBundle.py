"""
This code contains code for hybrid parallelism (HP) parameter server.
One Hybrid_PS contains two PSs - parameter server for front and rear.


"""
from termcolor import colored
from torchvision import datasets, transforms
from torch.autograd import Variable
from util import AverageMeter
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch, os,random, threading, model
import numpy as np


class Worker():

    NUM_WORKER = 0

    def __init__(self, batch_size, num_total_worker, args=None, local_rank=None):

        self.local_rank = Worker.NUM_WORKER if local_rank is None else local_rank
        self.batch_size = batch_size
        self.num_worker = num_total_worker
        self.args       = args
        self.__check_num_gpus__()
        Worker.NUM_WORKER += 1

        self.inter_sync = [i for i in range(args.world_size)]
        self.INTER_DP   = True if args.world_size > 1 else False
        self.MASTER     = True if self.local_rank == 0 else False
        self.SET_SYNC_CHANNEL = False

        if self.MASTER:
            self.COLLECTIVE_Q = []
            self.DISTRIBUTE_Q = []

            # Synchronization channel is required except for MASTER itself.
            for _ in range(self.num_worker - 1):
                self.COLLECTIVE_Q.append(mp.Queue())
                self.DISTRIBUTE_Q.append(mp.Queue())

        # Setting Average Meter
        self.collective = AverageMeter('collective', ':6.3f')
        self.distribute = AverageMeter('distribute', ':6.3f')
        self.intra_mp_send = AverageMeter('intra_mp', ':6.3f')
        self.sync_upload = AverageMeter('sync_upload', ':6.3f')
        self.sync_download = AverageMeter('sync_download', ':6.3f')
        self.inter_sync_comm = AverageMeter('inter_sync_comm', ':6.3f')
        self.comp_forward = AverageMeter('comp_forward', ':6.3f')
        self.comp_backprop = AverageMeter('comp_backprop', ':6.3f')

    def get_sync_channel(self):
        """
        This function should be invoked only for MASTER worker
        It will return sync channel (mp.Qs for collect / distribute gradients)
        :return:
        """

        if not self.MASTER:
            return -1, -1
        else:
            return self.COLLECTIVE_Q, self.DISTRIBUTE_Q

    def set_sync_channel(self, COLLECTIVE_Q, DISTRIBUTE_Q):
        """
        This function should be called before run( )
        :param COLLECTIVE_Q:
        :param DISTRIBUTE_Q:
        :return:
        """
        self.SET_SYNC_CHANNEL = True
        if self.MASTER:
            return
        self.COLLECTIVE_Q = COLLECTIVE_Q
        self.DISTRIBUTE_Q = DISTRIBUTE_Q

    def _check_set_sync_channel(self):

        if not self.SET_SYNC_CHANNEL and not self.MASTER:
            print(colored(' <ERROR!>', "red"),
                  colored(' set_sync_channel() should be called before run()\n','yellow'))
            exit()

    def run(self):

        self._check_set_sync_channel()
        self.gpu_rank   = self.local_rank
        model_name      = self.args.model
        Net             = getattr(model, model_name)

        # set INTER_DP
        if self.MASTER:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank),
                                    world_size = self.args.world_size)
            self.sync_group = dist.new_group(self.inter_sync)

        print("=====> (DP ONLY Bundle %d); worker start with GPU %d "% (self.local_rank, self.gpu_rank))

        # Prepare Training Data Set & Set SEED Value
        self._prepare_training()

        # Declare network model and allocate its memory on GPU
        self.net = Net().cuda(self.gpu_rank)

        # Define optimizer to be used in training
        optimizer   = optim.SGD(self.net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss().cuda(self.gpu_rank)


        for batch_idx, (data, target) in enumerate(self.train_loader):

            print("-----(BUNDLE %d) worker itr %d-----" % (self.local_rank, batch_idx))

            # load on gpu
            data = data.cuda(self.gpu_rank)
            target = target.cuda(self.gpu_rank)

            # feed forward
            self.comp_forward.tic()
            output = self.net(data)
            torch.cuda.synchronize(self.gpu_rank)
            self.comp_forward.toc()

            # calculate loss
            loss = criterion(output, target)

            # backpropagation
            self.comp_backprop.tic()
            loss.backward()
            torch.cuda.synchronize(self.gpu_rank)
            self.comp_backprop.toc()

            # Initialize PS (when ITR == 0)
            if batch_idx == 0 and self.MASTER:
                self._initialize_ps()

            # synchronize all gradients
            self._synchronization()

            # apply the update
            optimizer.step()

            del data, target
            if batch_idx == self.args.itr:
                return

    def _prepare_training(self):

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

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_worker,
            pin_memory  = True,
        )


        # For model parallelism; data shuffle matching
        SEED = 777 + self.local_rank
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        # Setting for forward
        torch.cuda.set_device(self.gpu_rank)

    def _initialize_ps(self):

        self.ps = []
        for name, layer in self.net.named_parameters():
            self.ps.append(layer.grad.data.cpu().detach())

    def _synchronization(self):

        if self.MASTER:

            # COLLECT
            threads = []
            for i in range(self.num_worker):

                # MASTER
                if i == 0:
                    t = threading.Thread(target=self._sync_local_update)
                else:
                    t = threading.Thread(target=self._sync_collect,
                                         args=(self.COLLECTIVE_Q[i-1],))

                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # INTER_SYNC
            if self.INTER_DP:
                for grad in self.ps:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sync_group)


            # DISTRIBUTE
            threads = []
            for i in range(self.num_worker):

                # MASTER
                if i == 0:
                    t = threading.Thread(target=self._sync_local_download)
                else:
                    t = threading.Thread(target=self._sync_distribute,
                                         args=(self.DISTRIBUTE_Q[i-1],))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        else:
            self._sync_upload()
            self._sync_download()

    def _sync_local_update(self):
        """
        FOR MASTER WORKER
        :return:
        """
        self.sync_upload.tic()
        for idx, layer in enumerate(self.net.parameters()):
            self.ps[idx] += layer.grad.cpu()
        self.sync_upload.toc()

    def _sync_local_download(self):
        """
        FOR MASTER WORKER
        :return:
        """
        self.sync_download.tic()
        for idx, layer in enumerate(self.net.parameters()):
            layer.grad.data.copy_(self.ps[idx])
        self.sync_download.toc()

    def _sync_upload(self):
        """
        FOR NONE-MASTER WORKER
        :return:
        """
        self.sync_upload.tic()
        for layer in self.net.parameters():
            self.COLLECTIVE_Q.put(layer.grad.cpu().detach())
        self.sync_upload.toc()

    def _sync_download(self):
        """
        FOR NONE-MASTER WORKER
        :return:
        """
        self.sync_download.tic()
        for layer in self.net.parameters():
            tmp = self.DISTRIBUTE_Q.get()
            layer.grad.data.copy_(tmp.data)
            del tmp
        self.sync_download.toc()

    def _sync_collect(self, queue):
        """
        SYNC COLLECTIVE IN MASTER-SIDE
        :return:
        """
        for param in self.ps:
            tmp = queue.get()
            param += tmp
            del tmp

    def _sync_distribute(self, queue):
        """
        SYNC DISTRIBUTE IN MASTER-SIDE
        :return:
        """
        for param in self.ps:
            queue.put(param)

    def __check_num_gpus__(self):

        # Test num_hp range check
        if torch.cuda.device_count() < self.num_worker :
            num_gpus = torch.cuda.device_count()
            print(colored(' <ERROR!>', "red"),
                  colored('Number of GPUs (%d) are insufficient to support hybrid_bundle with degree %d \n'% (num_gpus, self.num_worker),'yellow'),
                  colored('<ERROR!>', "red"),
                  colored('At least %d GPUs are required' % (self.num_worker), "yellow"))
            exit()

class Front_Worker(Worker):

    def __init__(self, batch_size, bundle_rank, num_front_worker, mp_forward_q, mp_backward_q, args=None):
        """
        :param batch_size:
        :param num_total_worker:
        :param args:
        """
        super(Front_Worker, self).__init__(batch_size, num_front_worker, args, bundle_rank)
        self.local_rank     = bundle_rank
        self.MP_FORWARD_Q   = mp_forward_q
        self.MP_BACKWARD_Q  = mp_backward_q
        self.MP_DEGREE      = 2
        self.args           = args
        self.MASTER         = True if self.local_rank == 0 else False

        if self.INTER_DP:
            self.inter_sync_front   = [self.MP_DEGREE * i for i in range(args.world_size)]
            self.inter_sync_rear    = [self.MP_DEGREE * i + 1 for i in range(args.world_size)]

    def run(self):
        self.gpu_rank   = self.local_rank * self.MP_DEGREE
        model_name      = self.args.model + "_front"
        Net             = getattr(model, model_name)

        if self.MASTER and self.INTER_DP:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank) * self.MP_DEGREE,
                                    world_size = self.args.world_size * self.MP_DEGREE)
            self.front_sync_group = dist.new_group(self.inter_sync_front)
            self.rear_sync_group  = dist.new_group(self.inter_sync_rear)
            self.sync_group = self.front_sync_group

        print("=====> (Bundle %d); front start with GPU %d "% (self.local_rank, self.gpu_rank))
        self._prepare_training()

        self.net = Net().cuda(self.gpu_rank)
        optimizer   = optim.SGD(self.net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            print("-----(BUNDLE %d) front itr %d-----" % (self.local_rank, batch_idx))
            # load on gpu
            data = data.cuda(self.gpu_rank)

            # feed forward
            output = self.net(data)

            # send feed forward output to rear
            self.MP_FORWARD_Q.put(output.cpu().detach())

            # receive backpropagation input from rear
            backward_input_tmp = self.MP_BACKWARD_Q.get()
            backward_input = backward_input_tmp.cuda(self.gpu_rank)

            # backpropagation
            output.backward(backward_input)

            # synchronization
            self._initialize_ps()
            self._synchronization()

            # update
            optimizer.step()

            # Memory garbage collection
            del backward_input_tmp

            if batch_idx == self.args.itr:
                return

class Rear_Worker(Worker):

    def __init__(self, batch_size, bundle_rank, num_rear_worker, mp_forward_q, mp_backward_q, args=None):
        super(Rear_Worker, self).__init__(batch_size, num_rear_worker, args, bundle_rank)
        self.local_rank     = bundle_rank
        self.MP_FORWARD_Q   = mp_forward_q
        self.MP_BACKWARD_Q  = mp_backward_q
        self.MP_DEGREE      = 2
        self.args           = args
        self.MASTER         = True if self.local_rank == 0 else False

        if self.INTER_DP:
            self.inter_sync_front   = [self.MP_DEGREE * i for i in range(args.world_size)]
            self.inter_sync_rear    = [self.MP_DEGREE * i + 1 for i in range(args.world_size)]

    def run(self):

        self.gpu_rank   = self.local_rank * self.MP_DEGREE + 1
        model_name      = self.args.model + "_rear"
        Net             = getattr(model, model_name)

        if self.MASTER and self.INTER_DP:
            dist.init_process_group(backend='gloo',
                                    init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                    rank = (self.args.rank) * self.MP_DEGREE + 1,
                                    world_size = self.args.world_size * self.MP_DEGREE)
            self.front_sync_group = dist.new_group(self.inter_sync_front)
            self.rear_sync_group  = dist.new_group(self.inter_sync_rear)
            self.sync_group = self.rear_sync_group

        print("=====> (Bundle %d); rear start with GPU %d "% (self.local_rank, self.gpu_rank))
        self._prepare_training()

        # Declare network model and allocate its memory on GPU
        self.net = Net().cuda(self.gpu_rank)

        # Define optimizer to be used in training
        optimizer   = optim.SGD(self.net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss().cuda(self.gpu_rank)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            print("-----(BUNDLE %d) rear itr %d-----" % (self.local_rank, batch_idx))

            # GET forward input (MP)
            forward_input_tmp = self.MP_FORWARD_Q.get()
            forward_input_tensor = forward_input_tmp.clone().cuda(self.gpu_rank)
            forward_input = Variable(forward_input_tensor, requires_grad=True).cuda(self.gpu_rank)
            del forward_input_tmp

            # FEED FORWARD
            forward_output = self.net(forward_input)
            target = target.cuda(self.gpu_rank)
            torch.cuda.synchronize(self.gpu_rank)

            loss = criterion(forward_output, target)
            loss.backward()
            torch.cuda.synchronize(self.gpu_rank)

            # SEND backward output (MP)
            self.MP_BACKWARD_Q.put(forward_input.grad.data)

            # synchronization
            self._initialize_ps()
            self._synchronization()

            # update
            optimizer.step()

            # memory garbage collection
            del forward_input, forward_output, target

            if batch_idx == self.args.itr:
                return

class Hybrid_Bundle():


    # << Caution >>
    # In spawned process, class variable will be initialized.
    # Do not refer class variable in new process but use object variable with self.
    NUM_HP = 0

    def __init__(self, batch_size, num_hp, args=None):

        self.num_hp     = num_hp
        self.batch_size = batch_size
        self.args       = args
        self.local_rank = Hybrid_Bundle.NUM_HP

        self.MP_DEGREE  = 2
        self.MASTER     = True if self.local_rank == 0 else False
        self.SET_SYNC_CHANNEL = False

        self.__check_num_gpus__()
        Hybrid_Bundle.NUM_HP += 1

        # inter HP node list (rank list)
        self.inter_sync_front   = [self.MP_DEGREE * i for i in range(args.world_size)]
        self.inter_sync_rear    = [self.MP_DEGREE * i + 1 for i in range(args.world_size)]

        # Create Qs
        self._create_bundle_mp_queue()

        # Create Front & Rear worker
        self.front_worker = Front_Worker(batch_size=batch_size,
                                         bundle_rank=self.local_rank,
                                         num_front_worker=num_hp,
                                         mp_forward_q=self.MP_FORWARD_Q,
                                         mp_backward_q=self.MP_BACKWARD_Q,
                                         args=args)

        self.rear_worker  = Rear_Worker(batch_size=batch_size,
                                        bundle_rank=self.local_rank,
                                        num_rear_worker=num_hp,
                                        mp_forward_q=self.MP_FORWARD_Q,
                                        mp_backward_q=self.MP_BACKWARD_Q,
                                        args=args)

    def get_sync_channel(self):

        if not self.MASTER:
            return -1, -1, -1, -1
        else:
            FRONT_COLLECTIVE_Q , FRONT_DISTRIBUTE_Q = self.front_worker.get_sync_channel()
            REAR_COLLECTIVE_Q , REAR_DISTRIBUTE_Q = self.rear_worker.get_sync_channel()
            return FRONT_COLLECTIVE_Q, FRONT_DISTRIBUTE_Q, REAR_COLLECTIVE_Q, REAR_DISTRIBUTE_Q

    def set_sync_channel(self, front_collective, front_distribute, rear_collective, rear_distribute):

        self.SET_SYNC_CHANNEL = True
        if self.MASTER:
            return

        self.front_worker.set_sync_channel(front_collective, front_distribute)
        self.rear_worker.set_sync_channel(rear_collective, rear_distribute)

    def run(self):
        """
        This function should be called only after get_sync_channel() and set_sync_channel() are set.
        If not, program will exit.
        :return:
        """

        self._check_set_sync_channel()
        processes = []
        print("=====> (Bundle %d) start"% self.local_rank)

        processes.append(mp.Process(target=self.front_worker.run))
        processes.append(mp.Process(target=self.rear_worker.run))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def _check_set_sync_channel(self):

        if not self.SET_SYNC_CHANNEL and not self.MASTER:
            print(colored(' <ERROR!>', "red"),
                  colored(' set_sync_channel() should be called before run()\n','yellow'))
            exit()

    def _create_bundle_mp_queue(self):

        self.MP_FORWARD_Q = mp.Queue()
        self.MP_BACKWARD_Q = mp.Queue()

    def _create_sync_queue(self):

        # Synchronous Q (Only Master Create)
        if self.MASTER:

            self.FRONT_COLLECTIVE_Q = []
            self.FRONT_DISTRIBUTE_Q = []
            self.REAR_COLLECTIVE_Q  = []
            self.REAR_DISTRIBUTE_Q  = []

            # MASTER will collect & distribute for synchronization
            for _ in range(self.num_hp-1):
                self.FRONT_COLLECTIVE_Q.append(mp.Queue())
                self.FRONT_DISTRIBUTE_Q.append(mp.Queue())
                self.REAR_COLLECTIVE_Q.append(mp.Queue())
                self.REAR_DISTRIBUTE_Q.append(mp.Queue())
        else:
            self.FRONT_COLLECTIVE_Q = None
            self.FRONT_DISTRIBUTE_Q = None
            self.REAR_COLLECTIVE_Q  = None
            self.REAR_DISTRIBUTE_Q  = None

    def __check_num_gpus__(self):

        # Test num_hp range check
        if torch.cuda.device_count() < self.num_hp * self.MP_DEGREE:
            num_gpus = torch.cuda.device_count()
            print(colored(' <ERROR!>', "red"),
                  colored('Number of GPUs (%d) are insufficient to support hybrid_bundle with degree %d \n'% (num_gpus, self.num_hp),'yellow'),
                  colored('<ERROR!>', "red"),
                  colored('At least %d GPUs are required' % (self.num_hp * self.MP_DEGREE), "yellow"))
            exit()

