"""
This code contains code for hybrid parallelism (HP) bundle.
One HP_Bundle contains two worker: Front_Worker and Rear_Worker
For Data Parallelism only, worker can be used.
For more detail of its usage, please refer to main.py

    - last update: 2019.10.15
    - E.Jubilee Yang
"""
import torch, threading, model, math
from termcolor import colored
from bundle_worker import Front, Rear
from util import AverageMeter, ProgressMeter
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import time


def _sync_collect_gradients(rank, ps, upload_q):
    for i, param in enumerate(ps):
        tmp = upload_q[rank].get()
        ps[i] += tmp.clone().detach()
        del tmp
    return


def _sync_distribute_gradients(rank, ps, download_q):
    for i, param in enumerate(ps):
        download_q[rank].put(param)


def _sync_init_t(rank, ps, upload_q):
    for i, param in enumerate(ps):
        tmp = upload_q[rank].get()
        ps[i] = tmp
        del tmp


def _sync_init(num_worker, ps, upload_q):
    ps_threads = []
    for rank in range(num_worker):
        t = threading.Thread(target=_sync_init_t, args=(rank, ps, upload_q))
        t.start()
        ps_threads.append(t)

    for t in ps_threads:
        t.join()


def _sync_collect_ps(num_worker, ps, upload_q):
    # collect gradients
    ps_threads = []
    for rank in range(num_worker):
        t = threading.Thread(target=_sync_collect_gradients,
                             args=(rank, ps, upload_q))
        t.start()
        ps_threads.append(t)

    for t in ps_threads:
        t.join()
    return


def _sync_distribute_ps(num_worker, ps, download_q):
    # distribute gradients over workers
    ps_threads = []
    for rank in range(num_worker):
        t = threading.Thread(target=_sync_distribute_gradients,
                             args=(rank, ps, download_q))
        t.start()
        ps_threads.append(t)

    for t in ps_threads:
        t.join()
    return


def _get_bundle_topology(shape):
    """
    :param shape: list describing bundle shape [#fronts, #rear]
    :return:
    """

    topology = []
    local_max_degree = torch.cuda.device_count()
    total_degree = shape[0] + shape[1]

    if total_degree <= local_max_degree:
        topology.append(shape)
        return topology

    elif shape[0] >= local_max_degree:
        topology.append([local_max_degree, 0])
        topology += _get_bundle_topology([shape[0] - local_max_degree, shape[1]])
        return topology

    elif (shape[0] is not 0) and (shape[0] <= local_max_degree):

        if (shape[1] > local_max_degree) and (total_degree % local_max_degree == 0):
            topology.append([shape[0], local_max_degree - shape[0]])
            topology += _get_bundle_topology([0, shape[1] - local_max_degree + shape[0]])
            return topology
        else:
            topology.append([shape[0], 0])
            topology += _get_bundle_topology([0, shape[1]])
            return topology

    elif shape[1] >= local_max_degree:
        topology.append([0, local_max_degree])
        topology += _get_bundle_topology([shape[0], shape[1] - local_max_degree])
        return topology


def _print_error(msg, is_exit=False):
    print(colored('\n-------------------- '
                  'ERROR MESSAGE PRINT '
                  '---------------------\n', "yellow"))
    print(colored(' <ERROR!>', "red"),
          colored(msg, 'yellow'))
    print(colored('\n---------------------- '
                  'EXIT THE PROGRAM '
                  '----------------------\n', "yellow"))
    if is_exit:
        exit()


class HP_BUNDLE():

    def __init__(self, shape, num_bundles, num_nodes, rank, args):
        """
        :param shape: shape of bundle
        :param num_bundles: (global) total number of bundles running in Bundle-based HP
        :param num_nodes:   (global) total number of nodes participating in this Bundle-based HP
        :param rank:        (global) rank of this node among the training cluster
        :param args:
        """

        self.shape = shape
        self.num_bundles = num_bundles
        self.num_nodes = num_nodes
        self.node_rank = rank
        self.args = args
        self.front_worker_bs = self.args.batch_size // self.shape[0]
        self.rear_worker_bs = self.args.batch_size // self.shape[1]

        # Get topology information & partial_bundle or not
        self._get_bundle_info()

        # bundles
        self.bundles = []


        # MICRO BUNDLE
        if self.IS_MICRO_BUNDLE:

            self.local_shape = self.topology[self.local_node_rank]

            rank = 0
            if self.local_shape[0] is not 0:

                self.front_hp_upQ = []
                self.front_hp_downQ = []

                batch_size = self.front_worker_bs * self.local_shape[0]
                self.bundles.append(Bundle(shape=[self.local_shape[0], 0],
                                           rank=rank,
                                           batch_size=batch_size,
                                           args=args))
                self.front_hp_downQ.append(mp.Queue())
                self.front_hp_upQ.append(mp.Queue())
                self.bundles[rank].set_front_hpQ(self.front_hp_upQ[0], self.front_hp_downQ[0])
                rank += 1

            elif self.local_shape[1] is not 0:

                self.rear_hp_upQ = []
                self.rear_hp_downQ = []

                batch_size = self.rear_worker_bs * self.local_shape[1]
                self.bundles.append(Bundle(shape=[0, self.local_shape[1]],
                                           rank=rank,
                                           batch_size=batch_size,
                                           args=args))
                self.rear_hp_downQ.append(mp.Queue())
                self.rear_hp_upQ.append(mp.Queue())
                self.bundles[rank].set_rear_hpQ(self.rear_hp_upQ[0], self.rear_hp_downQ[0])

        # monolithic bundle
        else:
            self.front_hp_downQ = []
            self.front_hp_upQ = []
            self.rear_hp_downQ = []
            self.rear_hp_upQ = []

            for rank in range(self.local_num_hp):
                self.bundles.append(Bundle(shape=self.shape,
                                           rank=rank,
                                           batch_size=self.batch_size,
                                           args=args,
                                           offset=rank * self.bundle_degree))
                self.front_hp_downQ.append(mp.Queue())
                self.front_hp_upQ.append(mp.Queue())
                self.bundles[rank].set_front_hpQ(uploadQ=self.front_hp_upQ[rank],
                                                 downloadQ=self.front_hp_downQ[rank])
                self.rear_hp_downQ.append(mp.Queue())
                self.rear_hp_upQ.append(mp.Queue())
                self.bundles[rank].set_rear_hpQ(uploadQ=self.rear_hp_upQ[rank],
                                                downloadQ=self.rear_hp_downQ[rank])

    def run(self):

        processes = []
        offset = 0
        # micro bundle
        if self.IS_MICRO_BUNDLE:

            # RUN HP_PS for front
            if self.local_shape[0] is not 0:
                p = mp.Process(target=self._hp_micro_front_ps,
                               args=(offset,))
                p.start()
                processes.append(p)
                offset += 1

            # RUN HP_PS for rear
            elif self.local_shape[1] is not 0:
                p = mp.Process(target=self._hp_micro_rear_ps,
                               args=(offset,))
                p.start()
                processes.append(p)

            # RUN bundles
            for bundle in self.bundles:
                p = mp.Process(target=bundle.run)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        # monolithic bundle
        else:
            p = mp.Process(target=self._hp_front_ps)
            p.start()
            processes.append(p)

            p = mp.Process(target=self._hp_rear_ps)
            p.start()
            processes.append(p)

            for bundle in self.bundles:
                p = mp.Process(target=bundle.run)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    def _get_bundle_info(self):
        """
        bundle_degree : number of sub bundles consisting of one bundle
        topology      : global topology
        :return:
        """

        self.topology = _get_bundle_topology(self.shape)
        self.bundle_degree = len(self.topology)
        self.num_gpus = torch.cuda.device_count()
        self.IS_MICRO_BUNDLE = False if self.bundle_degree == 1 else True
        # self.local_num_hp   = self.num_gpus // (self.topology[0][0] + self.topology[0][1])
        if self.IS_MICRO_BUNDLE:
            self.local_num_hp = 1
        else:
            local_max_num_hp = self.num_gpus // (self.topology[0][0] + self.topology[0][1])
            if self.args.num_bundle > local_max_num_hp * self.args.world_size:
                _print_error("Please check the running configuration ! \n"
                             "\tIn sufficient number of workers", True)

            num_hp = [local_max_num_hp] * self.args.world_size
            if self.args.num_bundle % local_max_num_hp is not 0:
                num_hp[self.args.num_bundle // local_max_num_hp] = self.args.num_hp % local_max_num_hp
            self.local_num_hp = num_hp[self.node_rank]

        if self.IS_MICRO_BUNDLE:
            self._get_micro_bundle_rank()
        else:
            self._get_monolithic_bundle_rank()

    def _get_micro_bundle_rank(self):
        """
        This function returns (global_bundle_rank, local_bundle_rank, topology, IS_LOCAL_MASTER)
        This function should be called in INTER_BUNDLE
        """

        self.local_node_rank = self.node_rank % self.bundle_degree
        self.bundle_rank = self.node_rank // self.bundle_degree

        if self.IS_MICRO_BUNDLE and self.bundle_degree * self.num_bundles > self.num_nodes:
            _print_error("This INTER_BUNDLE cannot run on this configuration \n" +
                         "\t  At least %d nodes are required to run one bundle \n" % (self.bundle_degree) +
                         "\t  Total %d gpus are required to run %d bundles" % (
                         self.bundle_degree * self.num_nodes, self.num_bundles), True)

        if self.bundle_degree is 1:
            _print_error("Use INTRA_BUNDLE instead of INTER_BUNDLE \n", True)

        self.front_intra_bundle_group = []
        self.rear_intra_bundle_group = []

        rank = 0
        for node in self.topology:

            if node[0] is not 0:
                self.front_intra_bundle_group.append(rank)
                rank += 1

            if node[1] is not 0:
                self.rear_intra_bundle_group.append(self.bundle_rank * self.bundle_degree + rank)
                rank += 1

        # number of HP_PS is bundle offset (not bundle_degree)
        # bundle degree denotes the number of worker required to run one bundle
        bundle_offset = len(self.front_intra_bundle_group) + len(self.rear_intra_bundle_group)
        self.front_intra_bundle_group = [i + bundle_offset * self.bundle_rank for i in self.front_intra_bundle_group]
        self.rear_intra_bundle_group = [i + bundle_offset * self.bundle_rank for i in self.rear_intra_bundle_group]
        self.mp_intra_bundle_group = [self.front_intra_bundle_group[0], self.rear_intra_bundle_group[0]]
        self.global_sync_front_dp = [i * bundle_offset for i in range(self.args.num_bundle)]
        self.global_sync_rear_dp = [i * bundle_offset + len(self.front_intra_bundle_group) for i in
                                    range(self.args.num_bundle)]

        # world_size
        self.world_size = bundle_offset * self.args.num_bundle

        # global rank of this micro-bundle
        self.global_rank = bundle_offset * self.bundle_rank + self.local_node_rank

    def _get_monolithic_bundle_rank(self):

        if self.local_num_hp * self.num_nodes < self.num_bundles:
            _print_error("Number of node is not sufficient to run this bundle", True)
        elif self.node_rank > self.num_bundles // self.local_num_hp:
            print(colored("<WARNING>", "yellow"),
                  "This node is not required in Bundle-based HP")
            exit()

        else:
            self.world_num_nodes = math.ceil(self.num_bundles / self.local_num_hp)
            self.world_size = self.world_num_nodes * 2
            self.batch_size = self.args.batch_size // self.num_bundles
            self.global_sync_front_dp = [2 * i for i in range(self.world_num_nodes)]
            self.global_sync_rear_dp = [2 * i + 1 for i in range(self.world_num_nodes)]
            self.front_intra_bundle_group = None
            self.rear_intra_bundle_group = None
            self.mp_intra_bundle_group = None

    def _hp_micro_front_ps(self, rank_offset):

        # Average Meter
        dist_sync = AverageMeter('dist_sync', ':6.3f')
        comm_forward = AverageMeter('comm_forward', ":6.3f")
        comm_backward = AverageMeter('comm_backward', ":6.3f")
        bundle_sync = AverageMeter('bundle_sync', ':6.3f')
        # Progress Meter
        progress = ProgressMeter(self.args.itr,
                                 'HP_FRONT',
                                 'white',
                                 dist_sync,
                                 comm_forward,
                                 comm_backward,
                                 bundle_sync)

        # declare dist process
        global_rank = self.global_rank + rank_offset
        dist.init_process_group(backend='gloo',
                                init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                rank=global_rank,
                                world_size=self.world_size)

        self.sync_front_group = dist.new_group(self.global_sync_front_dp)
        self.sync_rear_group = dist.new_group(self.global_sync_rear_dp)

        # For micro bundle
        self.dist_front_intra_bundle_group = dist.new_group(self.front_intra_bundle_group)
        self.dist_rear_intra_bundle_group = dist.new_group(self.rear_intra_bundle_group)
        self.dist_mp_intra_bundle_group = dist.new_group(self.mp_intra_bundle_group)

        # front ps
        self.front_ps = []
        front_model = getattr(model, self.args.model + "_front")
        front_model = front_model()
        for layer in front_model.parameters():
            self.front_ps.append(layer.grad)

        # front sender & receiver
        forward_mp_list = []
        backprop_mp_list = []

        # initialization
        input = torch.tensor(np.zeros([self.front_worker_bs, 3, 224, 224], np.float32))
        front_output = front_model(input)
        output_shape = [-1, front_output.shape[1], front_output.shape[2], front_output.shape[3]]
        bs = self.front_worker_bs * self.local_shape[0]
        backprop_shape = [bs] + output_shape[1:]
        backprop_tmp = torch.tensor(np.zeros(backprop_shape, np.float32))

        if self.local_node_rank == self.front_intra_bundle_group[0]:
            for rank in self.front_intra_bundle_group:
                bs = self.front_worker_bs * self.topology[rank][0]
                forward_shape = [bs] + output_shape[1:]
                forward_mp_list.append(torch.tensor(np.zeros(forward_shape, np.float32)))

        for itr in range(self.args.itr):

            # collect feedforward for mp
            feed_forward_tmp = self.front_hp_upQ[0].get()

            comm_forward.tic()
            # intra collect (dist.gather)
            dist.gather(tensor=feed_forward_tmp,
                        gather_list=forward_mp_list,
                        dst=self.front_intra_bundle_group[0],
                        group=self.dist_front_intra_bundle_group,
                        async_op=False)

            # send feed forward to rear server
            if self.local_node_rank == self.front_intra_bundle_group[0]:
                # create merged feed forward tensor
                merged_forward = torch.cat(forward_mp_list)
                # send feed forward tensor
                dist.send(tensor=merged_forward,
                          dst=self.mp_intra_bundle_group[1],
                          group=self.dist_mp_intra_bundle_group)
            comm_forward.toc()

            comm_backward.tic()
            # distribute backprop for mp
            if self.local_node_rank == self.front_intra_bundle_group[0]:
                # merged backpropagation tensor
                merged_backprop = torch.tensor(np.zeros(merged_forward.shape, np.float32))
                # receive from the representative rear server
                dist.recv(tensor=merged_backprop,
                          src=self.mp_intra_bundle_group[1],
                          group=self.dist_mp_intra_bundle_group)

                # local micro bundle batch size
                local_batch_size = self.front_worker_bs * self.local_shape[0]
                # split the backpropagation tensor into tensors list
                backprop_mp_list = list(torch.split(merged_backprop,
                                                    local_batch_size))

            # scatter the split backpropagation within intra-bundle
            dist.scatter(tensor=backprop_tmp,
                         scatter_list=backprop_mp_list,
                         src=self.front_intra_bundle_group[0],
                         group=self.dist_front_intra_bundle_group,
                         async_op=False)
            comm_backward.toc()

            # send the backprop tensor to Bundle
            self.front_hp_downQ[0].put(backprop_tmp)

            # collect data for sync
            if itr == 0:
                _sync_init(num_worker=1,
                           ps=self.front_ps,
                           upload_q=self.front_hp_upQ)
            else:
                _sync_collect_ps(num_worker=1,
                                 ps=self.front_ps,
                                 upload_q=self.front_hp_upQ)

            dist_sync.tic()
            # intra all reduce
            for grad in self.front_ps:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dist_front_intra_bundle_group)

            # inter all reduce
            if self.local_node_rank == self.front_intra_bundle_group[0]:
                for grad in self.front_ps:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sync_front_group)

            # intra distribute sync
            for idx, grad in enumerate(self.front_ps):
                dist.broadcast(tensor=self.front_ps[idx],
                               src=self.front_intra_bundle_group[0],
                               group=self.dist_front_intra_bundle_group)
            dist_sync.toc()

            # distribute data to bundle
            _sync_distribute_ps(num_worker=1,
                                ps=self.front_ps,
                                download_q=self.front_hp_downQ)

            progress.print_progress(itr+1)

        # WAIT
        time.sleep(5)
        return

    def _hp_micro_rear_ps(self, rank_offset):

        # Average Meter
        dist_sync = AverageMeter('dist_sync', ':6.3f')
        comm_forward = AverageMeter('comm_forward', ":6.3f")
        comm_backward = AverageMeter('comm_backward', ":6.3f")
        bundle_sync = AverageMeter('bundle_sync', ':6.3f')
        # Progress Meter
        progress = ProgressMeter(self.args.itr,
                                 'HP_FRONT',
                                 'white',
                                 dist_sync,
                                 comm_forward,
                                 comm_backward,
                                 bundle_sync)

        # declare dist process
        global_rank = self.global_rank + rank_offset
        dist.init_process_group(backend='gloo',
                                init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                rank=global_rank,
                                world_size=self.world_size)

        self.sync_front_group = dist.new_group(self.global_sync_front_dp)
        self.sync_rear_group = dist.new_group(self.global_sync_rear_dp)

        # For micro bundle
        self.dist_front_intra_bundle_group = dist.new_group(self.front_intra_bundle_group)
        self.dist_rear_intra_bundle_group = dist.new_group(self.rear_intra_bundle_group)
        self.dist_mp_intra_bundle_group = dist.new_group(self.mp_intra_bundle_group)

        # rear ps
        self.rear_ps = []
        rear_model = getattr(model, self.args.model + "_rear")
        rear_model = rear_model()

        # rear parameter server
        for layer in rear_model.parameters():
            self.rear_ps.append(layer.grad)

        # for the shape of the front model
        front_model = getattr(model, self.args.model + "_front")
        front_model = front_model()

        # rear sender & receiver
        forward_mp_list = []
        backprop_mp_list = []

        # initialization
        local_batch_size = self.rear_worker_bs * self.local_shape[1]
        input = torch.tensor(np.zeros([local_batch_size, 3, 224, 224], np.float32))
        front_output = front_model(input)
        rear_forward_input_tmp = torch.tensor(np.zeros(front_output.shape, np.float32))

        # initialization
        if self.local_node_rank == self.rear_intra_bundle_group[0]:
            output_shape = [self.args.batch_size,
                            front_output.shape[1],
                            front_output.shape[2],
                            front_output.shape[3]]
            forward_tensor = torch.tensor(np.zeros(output_shape, np.float32))
            for rank in self.rear_intra_bundle_group:
                bs = self.rear_worker_bs * self.topology[rank][1]
                backprop_shape = [bs] + output_shape[1:]
                backprop_mp_list.append(torch.tensor(np.zeros(backprop_shape, np.float32)))

        for itr in range(self.args.itr):

            comm_forward.tic()
            # get feed forward from dist for mp
            if self.local_node_rank == self.rear_intra_bundle_group[0]:
                # Receive forward tensors from front representative
                dist.recv(tensor=forward_tensor,
                          src=self.mp_intra_bundle_group[0],
                          group=self.dist_mp_intra_bundle_group)


                rear_local_bs = self.rear_worker_bs * self.local_shape[1]
                forward_scatter_list = list(torch.split(forward_tensor, rear_local_bs))

            # distribute forward tensors to rear workers
            dist.scatter(tensor=rear_forward_input_tmp,
                         scatter_list=forward_scatter_list,
                         src=self.rear_intra_bundle_group[0],
                         group=self.dist_rear_intra_bundle_group,
                         async_op=False)
            comm_forward.toc()


            # send the forward tensor to Bundle
            self.rear_hp_downQ[0].put(rear_forward_input_tmp)

            # collect back propagation tensor for mp
            back_propagation_tmp = self.rear_hp_upQ[0].get()

            comm_backward.tic()
            # intra collect (dist.gather)
            dist.gather(tensor=back_propagation_tmp,
                        gather_list=backprop_mp_list,
                        dst=self.rear_intra_bundle_group[0],
                        group=self.dist_rear_intra_bundle_group,
                        async_op=False)

            # send back propagation to front server
            if self.local_node_rank == self.rear_intra_bundle_group[0]:
                # create merged
                merged_backprop = torch.cat(backprop_mp_list)
                dist.send(tensor=merged_backprop,
                          dst=self.mp_intra_bundle_group[0],
                          group=self.dist_mp_intra_bundle_group)
            comm_backward.toc()

            # collect gradient data for sync
            if itr == 0:
                _sync_init(num_worker=1,
                           ps=self.rear_ps,
                           upload_q=self.rear_hp_upQ)
            else:
                _sync_collect_ps(num_worker=1,
                                 ps=self.rear_ps,
                                 upload_q=self.rear_hp_upQ)

            dist_sync.tic()
            # intra all reduce
            for grad in self.rear_ps:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dist_rear_intra_bundle_group)

            # inter all reduce
            if self.local_node_rank == self.rear_intra_bundle_group[0]:
                for grad in self.rear_ps:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sync_rear_group)

            # intra distribute sync
            for idx, grad in enumerate(self.rear_ps):
                dist.broadcast(tensor=self.rear_ps[idx],
                               src=self.rear_intra_bundle_group[0],
                               group=self.dist_rear_intra_bundle_group)
            dist_sync.toc()

            # distribute data to bundle
            _sync_distribute_ps(num_worker=1,
                                ps=self.rear_ps,
                                download_q=self.rear_hp_downQ)

            progress.print_progress(itr+1)

        # WAIT
        time.sleep(5)
        return

    def _hp_front_ps(self):

        # Average Meter
        dist_sync = AverageMeter('dist_sync', ':6.3f')
        comm_mp = AverageMeter('comm_mp', ":6.3f")
        # Progress Meter
        progress = ProgressMeter(self.args.itr,
                                      'HP_FRONT',
                                      'white',
                                      dist_sync,
                                      comm_mp)

        # declare dist process
        rank = self.args.rank * 2
        dist.init_process_group(backend='gloo',
                                init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                rank=rank,
                                world_size=self.world_size)

        self.sync_front_group = dist.new_group(self.global_sync_front_dp)
        self.sync_rear_group = dist.new_group(self.global_sync_rear_dp)

        # set front_ps & rear_ps
        self.front_ps = []
        front_model = getattr(model, self.args.model + "_front")
        front_model = front_model()
        for layer in front_model.parameters():
            self.front_ps.append(layer.grad)

        for itr in range(self.args.itr):

            # collect data from
            if itr == 0:
                _sync_init(num_worker=self.local_num_hp,
                           ps=self.front_ps,
                           upload_q=self.front_hp_upQ)
            else:
                _sync_collect_ps(num_worker=self.local_num_hp,
                                 ps=self.front_ps,
                                 upload_q=self.front_hp_upQ)

            for grad in self.front_ps:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sync_front_group)

            dist_sync.tic()
            # distribute data to bundle p
            _sync_distribute_ps(num_worker=self.local_num_hp,
                                ps=self.front_ps,
                                download_q=self.front_hp_downQ)
            dist_sync.toc()

            progress.print_progress(itr+1)

        # WAIT
        time.sleep(5)
        return

    def _hp_rear_ps(self):

        # Average Meter
        dist_sync = AverageMeter('dist_sync', ':6.3f')
        comm_mp = AverageMeter('comm_mp', ":6.3f")
        # Progress Meter
        progress = ProgressMeter(self.args.itr,
                                 'HP_REAR',
                                 'white',
                                 dist_sync,
                                 comm_mp)

        # declare dist process
        rank = self.args.rank * 2 + 1
        dist.init_process_group(backend='gloo',
                                init_method='tcp://%s:%s' % (self.args.IP, self.args.portNum),
                                rank=rank,
                                world_size=self.world_size)

        self.sync_front_group = dist.new_group([2 * i for i in range(self.world_num_nodes)])
        self.sync_rear_group = dist.new_group([2 * i + 1 for i in range(self.world_num_nodes)])

        # set front_ps & rear_ps
        self.rear_ps = []
        rear_model = getattr(model, self.args.model + "_rear")
        rear_model = rear_model()
        for layer in rear_model.parameters():
            self.rear_ps.append(layer.grad)

        for itr in range(self.args.itr):

            # collect data from
            if itr == 0:
                _sync_init(num_worker=self.local_num_hp,
                           ps=self.rear_ps,
                           upload_q=self.rear_hp_upQ)
            else:
                _sync_collect_ps(num_worker=self.local_num_hp,
                                 ps=self.rear_ps,
                                 upload_q=self.rear_hp_upQ)

            dist_sync.tic()
            for grad in self.rear_ps:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sync_rear_group)
            dist_sync.toc()

            # distribute data to bundle p
            _sync_distribute_ps(num_worker=self.local_num_hp,
                                ps=self.rear_ps,
                                download_q=self.rear_hp_downQ)

            progress.print_progress(itr+1)

        # WAIT
        time.sleep(5)
        return


class Bundle():
    """
    Bundle objects are crated
    only when all gpus are fit in the bundle size.
    i.e., there is no inter model parallelism.
    """

    def __init__(self, shape, rank, batch_size, args, offset=0):

        self.front_worker = []
        self.rear_worker = []

        # bundle rank
        self.rank = rank

        # front worker
        if shape[0] is not 0:
            front_bs = batch_size // shape[0]
            for i in range(shape[0]):
                self.front_worker.append(Front(rank=offset + i,
                                               batch_size=front_bs,
                                               args=args))
            offset += i

        # rear worker
        if shape[1] is not 0:
            rear_bs = batch_size // shape[1]
            for j in range(shape[1]):
                self.rear_worker.append(Rear(rank=offset + j,
                                             batch_size=rear_bs,
                                             args=args))
        self.ps = BundlePS(shape, batch_size, args)

        # set parameter server Q
        for rank, w in enumerate(self.front_worker):
            upQ, downQ = w.get_syncQ()
            self.ps.set_front_psQ(upQ, downQ, rank)

        for rank, w in enumerate(self.rear_worker):
            upQ, downQ = w.get_syncQ()
            self.ps.set_rear_psQ(upQ, downQ, rank)

    def set_front_hpQ(self, uploadQ, downloadQ):
        """
        only for monolithic Bundle
        :param uploadQ:
        :param downloadQ:
        :return:
        """
        self.ps.set_front_hpQ(uploadQ, downloadQ)

    def set_rear_hpQ(self, uploadQ, downloadQ):
        """
        only for monolithic Bundle
        :param uploadQ:
        :param downloadQ:
        :return:
        """
        self.ps.set_rear_hpQ(uploadQ, downloadQ)

    def run(self):

        processes = []

        p = mp.Process(target=self.ps.run)
        p.start()
        processes.append(p)

        for w in self.front_worker:
            p = mp.Process(target=w.run)
            p.start()
            processes.append(p)

        for w in self.rear_worker:
            p = mp.Process(target=w.run)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


class BundlePS():
    """
    Bundle Parameter Server
    - gathers front inference to send them to rear workers.
    - scatters front inference to rear workers
    - gathers rear backpropagation to send them to front workers.
    - scatters rear backpropagation to front workers
    - synchronize all gradients
    """

    def __init__(self, shape, batch_size, args):
        self.shape = shape
        self.bs = batch_size
        self.args = args

        # IS_MP is True only when Intra Bundle (for Inter Bundle, mp shape should be sublated
        self.IS_MP = False if (np.prod(shape) == 0) else True
        self.run = self._monolithic_ps_run if self.IS_MP else self._micro_ps_run
        self.init = self._monolithic_ps_init if self.IS_MP else self._micro_ps_init

        self.init()

    def _micro_ps_run(self):

        # Micro-ps manages front workers or rear workers only.
        if self.shape[1] == 0:
            front_ps = mp.Process(target=self._micro_front_ps)
            front_ps.start()
            front_ps.join()

        if self.shape[0] == 0:
            rear_ps = mp.Process(target=self._micro_rear_ps)
            rear_ps.start()
            rear_ps.join()

    def _monolithic_ps_run(self):

        ps = []
        front_ps = mp.Process(target=self._front_ps)
        front_ps.start()
        ps.append(front_ps)

        rear_ps = mp.Process(target=self._rear_ps)
        rear_ps.start()
        ps.append(rear_ps)

        for p in ps:
            p.join()

    def _micro_front_ps(self):

        for itr in range(self.args.itr):

            self._micro_mp_front_feedforward()
            self._micro_mp_front_backprop()

            # sync
            if itr == 0:
                _sync_init(num_worker=self.shape[0],
                           ps=self.front_ps,
                           upload_q=self.front_ps_uploadQ)
            else:
                # collect grads
                _sync_collect_ps(num_worker=self.shape[0],
                                 ps=self.front_ps,
                                 upload_q=self.front_ps_uploadQ)

            self._sync_ps(num_worker=self.shape[0],
                          ps=self.front_ps,
                          hp_upload_q=self.front_upload_hpQ,
                          hp_download_q=self.front_download_hpQ)

            # distribute graidnets
            _sync_distribute_ps(num_worker=self.shape[0],
                                ps=self.front_ps,
                                download_q=self.front_ps_downQ)

        # WAIT
        time.sleep(1)
        return

    def _micro_rear_ps(self):

        for itr in range(self.args.itr):

            self._micro_mp_rear_feedforward()
            self._micro_mp_rear_backprop()

            if itr == 0:
                _sync_init(num_worker=self.shape[1],
                           ps=self.rear_ps,
                           upload_q=self.rear_ps_uploadQ)
            else:
                # collect grads
                _sync_collect_ps(num_worker=self.shape[1],
                                 ps=self.rear_ps,
                                 upload_q=self.rear_ps_uploadQ)

            self._sync_ps(num_worker=self.shape[1],
                          ps=self.rear_ps,
                          hp_upload_q=self.rear_upload_hpQ,
                          hp_download_q=self.rear_download_hpQ)

            # distribute gradients
            _sync_distribute_ps(num_worker=self.shape[1],
                                ps=self.rear_ps,
                                download_q=self.rear_ps_downQ)
        # WAIT
        time.sleep(1)
        return

    def _front_ps(self):

        for itr in range(self.args.itr):

            self._mp_front_feedforward()
            self._mp_front_backprop()

            # sync
            if itr == 0:
                _sync_init(num_worker=self.shape[0],
                           ps=self.front_ps,
                           upload_q=self.front_ps_uploadQ)
            else:
                # collect grads
                _sync_collect_ps(num_worker=self.shape[0],
                                 ps=self.front_ps,
                                 upload_q=self.front_ps_uploadQ)

            self._sync_ps(num_worker=self.shape[0],
                          ps=self.front_ps,
                          hp_upload_q=self.front_upload_hpQ,
                          hp_download_q=self.front_download_hpQ)

            # distribute graidnets
            _sync_distribute_ps(num_worker=self.shape[0],
                                ps=self.front_ps,
                                download_q=self.front_ps_downQ)

        # WAIT
        time.sleep(1)
        return

    def _rear_ps(self):

        for itr in range(self.args.itr):

            self._mp_rear_feedforward()
            self._mp_rear_backprop()

            if itr == 0:
                _sync_init(num_worker=self.shape[1],
                           ps=self.rear_ps,
                           upload_q=self.rear_ps_uploadQ)
            else:
                # collect grads
                _sync_collect_ps(num_worker=self.shape[1],
                                 ps=self.rear_ps,
                                 upload_q=self.rear_ps_uploadQ)

            self._sync_ps(num_worker=self.shape[1],
                          ps=self.rear_ps,
                          hp_upload_q=self.rear_upload_hpQ,
                          hp_download_q=self.rear_download_hpQ)

            # distribute gradients
            _sync_distribute_ps(num_worker=self.shape[1],
                                ps=self.rear_ps,
                                download_q=self.rear_ps_downQ)
        # WAIT
        time.sleep(1)
        return

    def _micro_mp_rear_feedforward(self):

        forward_tmp = self.rear_download_hpQ.get()

        # distribute tensors
        self.rear_tmp = list(torch.split(forward_tmp, self.rear_bs, 0))

        # distribute feedforward
        forwards = []
        for rank in range(self.shape[1]):
            t = threading.Thread(target=self._distribute,
                                 args=(self.rear_ps_downQ,
                                       self.rear_tmp,
                                       rank))
            t.start()
            forwards.append(t)

        for t in forwards:
            t.join()

    def _micro_mp_rear_backprop(self):

        # get backprop
        backprops = []
        for rank in range(self.shape[1]):
            t = threading.Thread(target=self._backprop_collect, args=(rank,))
            t.start()
            backprops.append(t)

        for t in backprops:
            t.join()

        # concat all backprops
        backprop_tmp = torch.cat(self.rear_tmp, 0)

        self.rear_upload_hpQ.put(backprop_tmp)

    def _micro_mp_front_feedforward(self):

        # get forwards
        forwards = []
        for rank in range(self.shape[0]):
            t = threading.Thread(target=self._collect,
                                 args=(self.front_ps_uploadQ,
                                       self.front_tmp,
                                       rank))
            t.start()
            forwards.append(t)

        for t in forwards:
            t.join()

        # upload forward to hp-ps
        # to send forward to rear master and distribute them.
        forward_tmp = torch.cat(self.front_tmp, 0)
        self.front_upload_hpQ.put(forward_tmp)

    def _micro_mp_front_backprop(self):

        backprop_tmp = self.front_download_hpQ.get()

        # distribute tensors
        self.front_tmp = list(torch.split(backprop_tmp, self.front_bs, 0))

        # distribute backpropagation
        backprops = []
        for rank in range(self.shape[0]):
            t = threading.Thread(target=self._backprop_distribute, args=(rank,))
            t.start()
            backprops.append(t)

        for t in backprops:
            t.join()

    def _mp_front_feedforward(self):

        # get forwards
        forwards = []
        for rank in range(self.shape[0]):
            t = threading.Thread(target=self._collect,
                                 args=(self.front_ps_uploadQ,
                                       self.front_tmp,
                                       rank))
            t.start()
            forwards.append(t)

        for t in forwards:
            t.join()

        # concat all inputs
        forward_tmp = torch.cat(self.front_tmp, 0)
        self.mp_feedforward_q.put(forward_tmp)

    def _mp_rear_feedforward(self):

        forward_tmp = self.mp_feedforward_q.get()

        # distribute tensors
        self.rear_tmp = list(torch.split(forward_tmp, self.rear_bs, 0))

        # distribute feedforward
        forwards = []
        for rank in range(self.shape[1]):
            t = threading.Thread(target=self._distribute,
                                 args=(self.rear_ps_downQ,
                                       self.rear_tmp,
                                       rank))
            t.start()
            forwards.append(t)

        for t in forwards:
            t.join()

    def _mp_rear_backprop(self):

        # get backprop
        backprops = []
        for rank in range(self.shape[1]):
            t = threading.Thread(target=self._backprop_collect, args=(rank,))
            t.start()
            backprops.append(t)

        for t in backprops:
            t.join()

        # concat all backprops
        backprop_tmp = torch.cat(self.rear_tmp, 0)

        self.mp_backprop_q.put(backprop_tmp)

    def _mp_front_backprop(self):

        backprop_tmp = self.mp_backprop_q.get()

        # distribute tensors
        self.front_tmp = list(torch.split(backprop_tmp, self.front_bs, 0))

        # distribute backpropagation
        backprops = []
        for rank in range(self.shape[0]):
            t = threading.Thread(target=self._backprop_distribute, args=(rank,))
            t.start()
            backprops.append(t)

        for t in backprops:
            t.join()

    @staticmethod
    def _collect(queue, memory, rank):

        tensor = queue[rank].get()
        memory[rank] = tensor.clone().detach()
        del tensor
        return

    @staticmethod
    def _distribute(queue, memory, rank):
        queue[rank].put(memory[rank])

    def _sync_ps(self, num_worker, ps, hp_upload_q, hp_download_q):

        # upload to hp-ps
        for grad in ps:
            hp_upload_q.put(grad)

        # download from hp-ps
        for idx, grad in enumerate(ps):
            ps[idx] = hp_download_q.get()
        return

    def _forward_distribute(self, rank):
        self.rear_ps_downQ[rank].put(self.rear_tmp[rank])

    def _backprop_distribute(self, rank):
        self.front_ps_downQ[rank].put(self.front_tmp[rank])

    def _backprop_collect(self, rank):
        tensor = self.rear_ps_uploadQ[rank].get()
        self.rear_tmp[rank] = tensor.clone().detach()
        del tensor

    def _split_backprop_for_front(self):
        # concat all backprops
        backprop_tmp = torch.cat(self.rear_tmp, 0)

        # distribute tensors
        self.front_tmp = list(torch.split(backprop_tmp, self.front_bs, 0))

    def _split_forward_for_rear(self):
        # concat all inputs
        forward_tmp = torch.cat(self.front_tmp, 0)

        # distribute tensors
        self.rear_tmp = list(torch.split(forward_tmp, self.rear_bs, 0))

    def _monolithic_ps_init(self):
        """
        this function is called only when monolithic-bundle is used
        :return:
        """

        # batch size
        self.front_bs = self.bs // self.shape[0]
        self.rear_bs = self.bs // self.shape[1]

        # tmp memory
        self.front_tmp = [None] * self.shape[0]
        self.rear_tmp = [None] * self.shape[1]

        # ps queu
        self.front_ps_uploadQ = [None] * self.shape[0]
        self.front_ps_downQ = [None] * self.shape[0]
        self.rear_ps_uploadQ = [None] * self.shape[1]
        self.rear_ps_downQ = [None] * self.shape[1]

        # offset for rear worker rank
        self.rear_offset = self.shape[0]

        # set front_ps & rear_ps
        self.front_ps = []
        front_model = getattr(model, self.args.model + "_front")
        front_model = front_model()
        for layer in front_model.parameters():
            self.front_ps.append(layer.grad)

        self.rear_ps = []
        rear_model = getattr(model, self.args.model + "_rear")
        rear_model = rear_model()
        for layer in rear_model.parameters():
            self.rear_ps.append(layer.grad)

        self.mp_feedforward_q = mp.Queue()
        self.mp_backprop_q = mp.Queue()

    # NOT YET DEFINED
    def _micro_ps_init(self):
        """
        this function is called only when micro-bundle is used
        :return:
        """

        if self.shape[0] is not 0:
            self.front_bs = self.bs // self.shape[0]
            self.front_tmp = [None] * self.shape[0]
            self.front_ps_uploadQ = [None] * self.shape[0]
            self.front_ps_downQ = [None] * self.shape[0]
            self.front_ps = []
            front_model = getattr(model, self.args.model + "_front")
            front_model = front_model()
            for layer in front_model.parameters():
                self.front_ps.append(layer.grad)

        if self.shape[1] is not 0:
            self.rear_bs = self.bs // self.shape[1]
            self.rear_tmp = [None] * self.shape[1]
            self.rear_ps_uploadQ = [None] * self.shape[1]
            self.rear_ps_downQ = [None] * self.shape[1]
            self.rear_ps = []
            rear_model = getattr(model, self.args.model + "_rear")
            rear_model = rear_model()
            for layer in rear_model.parameters():
                self.rear_ps.append(layer.grad)

    def set_front_psQ(self, uploadQ, downloadQ, rank):

        self.front_ps_uploadQ[rank] = uploadQ
        self.front_ps_downQ[rank] = downloadQ

    def set_rear_psQ(self, uploadQ, downloadQ, rank):

        self.rear_ps_uploadQ[rank] = uploadQ
        self.rear_ps_downQ[rank] = downloadQ

    def set_front_hpQ(self, uploadQ, downloadQ):
        self.front_upload_hpQ = uploadQ
        self.front_download_hpQ = downloadQ

    def set_rear_hpQ(self, uploadQ, downloadQ):
        self.rear_upload_hpQ = uploadQ
        self.rear_download_hpQ = downloadQ
