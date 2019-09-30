#!/usr/bin/env python
import time

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def tic(self):
        self.t_tic = time.time()

    def toc(self):
        self.update(time.time() - self.t_tic)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

    def __init__(self, itr, prefix="", *meters):

        self.progress_fmtstr = self._get_batch_fmtstr(itr) + ']'
        self.meters = meters
        self.prefix = prefix

    def print_progress(self, itr):

        entries = [self.prefix + self.progress_fmtstr.format(itr)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_progress_fmtstr(self, itr):

        num_digits = len(str(itr // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(itr) + ']'
