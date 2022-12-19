import math

from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, pow, max_iter, min_lrs, last_epoch=-1, warmup=0):
        self.pow = pow
        self.max_iter = max_iter
        if not isinstance(min_lrs, list) and not isinstance(min_lrs, tuple):
            self.min_lrs = [min_lrs] * len(optimizer.param_groups)
        assert isinstance(warmup, int), 'Wrong warmup type, got {}'.format(type(warmup))
        self.warmup = max(warmup, 0)

        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [(base_lr / self.warmup) * (self.last_epoch + 1) for base_lr in self.base_lrs]

        if self.last_epoch < self.max_iter:
            coffe = math.pow(1 - (self.last_epoch - self.warmup) / (self.max_iter - self.warmup), self.pow)
        else:
            coffe = 0

        return [min_lrs + (base_lr - min_lrs) * coffe for base_lr, min_lrs in zip(self.base_lrs, self.min_lrs)]
