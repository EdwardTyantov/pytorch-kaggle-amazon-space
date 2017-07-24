#-*- coding: utf8 -*-
#credits to https://github.com/pytorch/examples/blob/master/imagenet/main.py
import shutil, time, logging
import torch
import torch.optim
import numpy as np
import visdom, copy
from datetime import datetime
from collections import defaultdict
from generic_models.yellowfin import YFOptimizer


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class VisdomMonitor(object):
    def __init__(self, prefix=None, server='http://localhost', port=8097):
        self.__prefix = prefix or datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.__vis = visdom.Visdom(server=server, port=port)
        self.__metrics = defaultdict(lambda :defaultdict(list))
        self.__win_dict = {}
        self.__opts = self._init_opts()

    def _init_opts(self):
        opts = dict(legend=['Train', 'Validate'])
        return opts

    def __add(self, name, value, type):
        self.__metrics[type][name].append(value)

    def _add_val_performance(self, name, value):
        self.__add(name, value, type='val')

    def _add_train_performance(self, name, value):
        self.__add(name, value, type='train')

    def add_performance(self, metric_name, train_value, val_value):
        self._add_train_performance(metric_name, train_value )
        self._add_val_performance(metric_name, val_value)
        self.plot(metric_name)

    def plot(self, metric_name):
        current_win = self.__win_dict.get(metric_name, None)
        train_values = self.__metrics['train'][metric_name]
        val_values = self.__metrics['val'][metric_name]
        epochs = max(len(train_values), len(val_values))
        values_for_plot = np.column_stack((np.array(train_values), np.array(val_values)))
        opts = copy.deepcopy(self.__opts)
        opts.update(dict(title='%s\ntrain/val %s' % (self.__prefix, metric_name)))
        win = self.__vis.line(Y=values_for_plot, X=np.arange(epochs), opts=opts, win=current_win)

        if current_win is None:
            self.__win_dict[metric_name] = win


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate_by_schedule(config, optimizer, epoch, decrease_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 1/decrease_rate every 10 epochs"""
    if not isinstance(optimizer, torch.optim.SGD):
        return
    #lr = config.lr * (0.1 ** (epoch // 10))
    if epoch and epoch % 10 == 0:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] *= decrease_rate
            logger.info('Setting learning layer=i, rate=%.6f', i, param_group['lr'])


class PlateauScheduler(object):
    """Sets the lr to the initial LR decayed by 1/decrease_rate, when not improving for max_stops epochs"""
    def __init__(self, optimizer, patience, early_stop_n, decrease_rate=0.1, eps=1e-5,
                 warm_up_epochs=None, best_score=None):
        self.optimizer = optimizer
        if not isinstance(optimizer, (torch.optim.SGD, YFOptimizer)):
            raise TypeError
        self.patience = patience
        self.early_stop_n = early_stop_n
        self.decrease_rate = decrease_rate
        self.eps = eps
        self.warm_up_epochs = warm_up_epochs
        self.__lr_changed = 0
        self.__early_stop_counter = 0
        self.__best_score = best_score
        self.__descrease_times = 0
        self.__warm_up = self.__has_warm_up(optimizer)

    def __has_warm_up(self, optimizer):
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] != param_group['after_warmup_lr']:
                logger.info('Optimizer has warm-up stage')
                return True

    def step(self, epoch, score):
        adjusted, to_break = False, False

        prev_best_score = self.__best_score or -1
        is_best = self.__best_score is None or score < self.__best_score - self.eps
        self.__best_score = self.__best_score is not None and min(score, self.__best_score) or score
        if is_best:
            logger.info('Current model is best by val score %.5f < %.5f' % (self.__best_score, prev_best_score))
            self.__early_stop_counter = 0
        else:
            self.__early_stop_counter += 1
            if self.__early_stop_counter >= self.early_stop_n:
                logger.info('Early stopping, regress for %d iterations', self.__early_stop_counter)
                to_break = True
        logger.info('early_stop_counter: %d', self.__early_stop_counter)

        if (self.warm_up_epochs and self.__descrease_times == 0 and self.__warm_up and epoch >= self.warm_up_epochs - 1 ) or \
                (self.__lr_changed <= epoch - self.patience and \
                (self.__early_stop_counter is not None and self.patience and self.__early_stop_counter >= self.patience)):
            self.__lr_changed = epoch
            for param_group in self.optimizer.param_groups:
                if self.__descrease_times == 0 and self.__warm_up:
                    param_group['lr'] = param_group['after_warmup_lr']
                else:
                    param_group['lr'] = param_group['lr'] * self.decrease_rate
                logger.info('Setting for group learning rate=%.8f, epoch=%d', param_group['lr'], self.__lr_changed)
            adjusted = True
            self.__descrease_times += 1

        return adjusted, to_break, is_best


def init_optimizer(model, config, exact_layers=None):
    """param 'exact_layers' specifies which parameters of the model to train, None - all,
       else - list of layers with a multiplier (optional) for LR schedule"""
    opt_type = config.optimizer
    if exact_layers:
        logger.info('Learning exact layers, number=%d', len(exact_layers))
        parameters = []
        for i, layer in enumerate(exact_layers):
            if isinstance(layer, tuple) and len(layer) == 2:
                layer, multiplier = layer
                init_multiplier = 1
            elif isinstance(layer, tuple) and len(layer) == 3:
                layer, init_multiplier, multiplier = layer
            else:
                multiplier = 1
                init_multiplier = 1
            lr = config.lr * multiplier
            init_lr = config.lr * multiplier * init_multiplier
            logger.info('Layer=%d, lr=%.5f', i, init_lr)
            parameters.append({'params': layer.parameters(), 'lr': init_lr, 'after_warmup_lr': lr})
    else:
        logger.info('Optimizing all parameters, lr=%.5f', config.lr)
        parameters = model.parameters()

    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(parameters, config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif opt_type == 'yf':
        optimizer = YFOptimizer(parameters, config.lr, mu=config.momentum, weight_decay=config.weight_decay,
                                clip_thresh=0.1)
    else:
        raise TypeError, 'Unknown optimizer type=%s' % (opt_type, )
    return optimizer


def save_checkpoint(state, epoch, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
        shutil.copyfile(filename, best_filename + '-%d' % epoch)


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    predictions = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i and i % 50 == 0) or i == len(train_loader) - 1:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=predictions))

    return losses.avg


def compute_f2(output, target):
    true_and_pred = target * output

    ttp_sum = torch.sum(true_and_pred, 1)
    tpred_sum = torch.sum(output, 1)
    ttrue_sum = torch.sum(target, 1)

    tprecision = ttp_sum / tpred_sum
    trecall = ttp_sum / ttrue_sum
    f2 = ((1 + 4) * tprecision * trecall) / (4 * tprecision + trecall)

    return f2


def validate(val_loader, model, criterion, activation=torch.sigmoid):
    logger.info('Validating model')
    batch_time = AverageMeter()
    losses = AverageMeter()
    f2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute f2
        f2 = compute_f2(activation(output), target_var).mean()
        f2s.update(f2.data[0], input.size(0))

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Test: [{0}/{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.avg:.5f}\t'
          'F2: {f2s.avg}\t'.format(
           len(val_loader), batch_time=batch_time, loss=losses, f2s=f2s))

    return losses.avg


def get_outputs(loader, model, activation):
    model.eval()
    outputs, targets = [], []
    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        if activation is not None:
            output = activation(output)
        outputs.extend(output.cpu().data)
        targets.extend(target)
    return outputs, targets


def test_model(test_loader, model, activation=None):
    logger.info('Testing')
    model.eval()

    names, results = [], []
    for i, (input, name_batch) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        output = model(input_var)
        if activation is not None:
            output = activation(output)

        names.extend(name_batch)
        results.extend(output.cpu())
        if i and i % 20 == 0:
            logger.info('Batch %d',i)

    return names, results
