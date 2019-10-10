from util import AverageMeter, ProgressMeter
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn    as nn
import numpy       as np
import os, torch, random, model
import time


class Worker():

    def __init__(self, rank, batch_size, args, seed=None):

        self.local_rank = rank
        self.batch_size = batch_size
        self.args       = args
        self.up_Q       = mp.Queue()
        self.down_Q     = mp.Queue()
        self.seed       = seed

        # setting average meter

    def get_syncQ(self):
        return self.up_Q, self.down_Q

    def run(self):

        model_name  = self.args.model
        Net         = getattr(model, model_name)
        self.gpu_rank= self.local_rank

        # Prepare Training data set & set seed value
        self._prepare_training()

        # Declare network model and allocate its memory on GPU
        self.net    = Net().cuda(self.gpu_rank)

        optimizer   = optim.SGD(self.net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)

        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss().cuda(self.gpu_rank)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            # load on gpu
            data = data.cuda(self.gpu_rank)
            target = target.cuda(self.gpu_rank)

            # feed forward
            output = self.net(data)
            torch.cuda.synchronize(self.gpu_rank)

            # calculate loss
            loss = criterion(output, target)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize(self.gpu_rank)

            # synchronize all gradients
            self._synchronization()

            # apply the update
            optimizer.step()

            del data, target

            if batch_idx == (self.args.itr - 1):
                time.sleep(3)
                print("worker return")
                return

    def _synchronization(self):
        self._sync_upload()
        self._sync_download()

    def _sync_upload(self):

        for layer in self.net.parameters():
            grad = layer.grad.cpu().detach()
            self.up_Q.put(grad)

    def _sync_download(self):

        for layer in self.net.parameters():
            tmp = self.down_Q.get()
            layer.grad.data.copy_(tmp)
            del tmp

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
            num_workers = 1,
            pin_memory  = True,
        )

        # For model parallelism; data shuffle matching
        SEED = self.seed if self.seed is not None else 7777
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        # Setting for forward
        torch.cuda.set_device(self.local_rank)

    def _upload_to_ps(self, data):
        self.up_Q.put(data)

    def _download_from_ps(self):

        tmp = self.down_Q.get()
        return_val = tmp.clone().detach()
        del tmp
        return return_val

class Front(Worker):

    def __init__(self, rank, batch_size, args, seed=None):
        
        super(Front, self).__init__(rank, batch_size, args, seed)
        # setting additional average meter
        # update progress meter

    def run(self):

        self.gpu_rank   = self.local_rank
        model_name      = self.args.model + "_front"
        Net             = getattr(model, model_name)

        self._prepare_training()
        self.net = Net().cuda(self.gpu_rank)
        optimizer   = optim.SGD(self.net.parameters(),
                                lr       = self.args.lr,
                                momentum = self.args.momentum,
                                weight_decay = self.args.weight_decay)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            # load on gpu
            data = data.cuda(self.gpu_rank)
            optimizer.zero_grad()

            # feed forward
            output = self.net(data)
            print("<front> feed forward")

            # upload feed forward output to ps (to send rear worker the outcome)
            self._upload_feedforward(output)
            print("<front> upload forward")

            # download backprop input from ps
            self._download_backprop()
            print("<front> download backprop")

            # backpropagation
            output.backward(self.backprop_input)
            print("<front> backpropagation")

            # synchronization
            self._synchronization()
            print("<front> synchronization")

            # update
            optimizer.step()
            del data, output

            if batch_idx == (self.args.itr -1):
                return

    def _upload_feedforward(self, output):
        gpu_to_cpu = output.cpu().detach()
        self._upload_to_ps(gpu_to_cpu)

    def _download_backprop(self):
        cpu_to_gpu = self._download_from_ps()
        self.backprop_input = cpu_to_gpu.cuda(self.gpu_rank)


class Rear(Worker):

    def __init__(self, rank, batch_size, args, seed=None):
        
        super(Rear, self).__init__(rank, batch_size, args, seed)
        # setting additional average meter
        # update progress meter

    def run(self):

        self.gpu_rank   = self.local_rank
        model_name      = self.args.model + "_rear"
        Net             = getattr(model, model_name)

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

            # load target data on GPU memory
            optimizer.zero_grad()
            target = target.cuda(self.gpu_rank)

            # Receive feed forward input from ps
            self._download_feedforward()
            print("<rear> download feed forward")

            # feed forward
            forward_output = self.net(self.forward_input)
            print("<rear> feed forward")

            # loss
            loss = criterion(forward_output, target)
            print("<rear> calculate loss")

            # backpropagation
            loss.backward()
            print("<rear> backpropagation ")

            # upload backpropagation result
            self._upload_backprop()
            print("<rear> upload_backprop ")

            # synchronization
            self._synchronization()
            print("<rear> synchronization ")

            # update
            optimizer.step()

            # garbage collection
            del forward_output, target

            if batch_idx == (self.args.itr - 1):
                print("rear worker return")
                return

    def _download_feedforward(self):

        tmp = self._download_from_ps()
        tmp = tmp.cuda(self.gpu_rank)
        self.forward_input = Variable(tmp, requires_grad=True).cuda(self.gpu_rank)

    def _upload_backprop(self):

        self._upload_to_ps(self.forward_input.grad.data.cpu().detach())



