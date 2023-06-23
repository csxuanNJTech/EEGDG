import math


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr
    if bl:
        lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))
    else:
        lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args.max_epochs))) / 1.01)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inv_lr_scheduler(optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
     https://blog.csdn.net/bc521bc/article/details/85864555"""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
