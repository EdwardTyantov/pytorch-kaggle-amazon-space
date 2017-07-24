#-*- coding: utf8 -*-
import sys, random, os, logging, glob, traceback, json
from itertools import izip
from collections import defaultdict
from optparse import OptionParser
import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data.sampler import WeightedRandomSampler
import torch, torch.nn as nn, torch.nn.parallel, torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import boilerplate
from folder import ImageFolder, ImageTestFolder
from __init__ import RESULT_DIR, NUM_CLASSES, SPLIT_FILE, HOLDOUT_FILE, BLACKLIST_FILE
import model, transform_rules
from folder import default_loader, tif_loader, tif_index_loader, mix_loader, jpg_cv2_loader, jpg_nir_loader


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, model_func, transform_func, label_file, train_dir, test_dir, seed=0,
                 shard='', pre_trained=True):
        self.__model_func = model_func
        self.__transform_func = transform_func
        self.__label_file = label_file
        self.__train_dir = train_dir
        self.__test_dir = test_dir
        self.__seed = seed
        self.__pre_trained = pre_trained
        res_dir = RESULT_DIR.format(shard)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        # modify logs for sharding (parallel models on different GPUs)
        if shard != '':
            fh = logging.FileHandler(os.path.join(res_dir, 'application.log'))
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            logger.handlers[0] = fh

        self.__checkpoint_file = os.path.join(res_dir, 'checkpoint{0}.pth.tar')
        self.__best_model_filename = os.path.join(res_dir, 'model_best{0}.pth.tar')
        self.__output_file = os.path.join(res_dir, 'submission{0}.csv')
        self.__detailed_output_file = os.path.join(res_dir, 'detailed_submission{0}.txt')
        self.__holdout_output_file = os.path.join(res_dir, 'detailed_holdout{0}.txt')
        self.__valid_output_file = os.path.join(res_dir, 'detailed_valid{0}.txt')
        self._model, self.__layers_to_optimize = None, None
        self.monitor = boilerplate.VisdomMonitor(port=80)
        self.__cur_fold = ''

    def __rm_prev_checkpoints(self):
        for fname in glob.glob(self.__best_model_filename.format(self.__cur_fold) + '-*'):
            os.remove(fname)

    def _init_model(self):
        logger.info('Initing model')
        model, layers = self.__model_func(num_classes=NUM_CLASSES, pretrained=self.__pre_trained)
        if torch.has_cudnn:
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)
            model = model.cuda()
            cudnn.benchmark = True
        else:
            raise RuntimeError, 'The model in CPU mode, the code is designed for cuda only'
        self._model = model
        self.__layers_to_optimize = layers

    def _init_checkpoint(self, optimizer, config):
        start_epoch = 0
        best_score = None

        if config.from_checkpoint:
            checkpoint = boilerplate.load_checkpoint(self.__checkpoint_file.format(self.__cur_fold))
            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_prec1']
            self._model.load_state_dict(checkpoint['state_dict'])
            # Sometimes cause error b/c of multiple param groups, pytorch bug ?
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Resuming from checkpoint epoch=%d, score=%.5f' % (start_epoch, best_score or -1))

        return start_epoch, best_score

    def _save_checkpoint(self, config, optimizer, epoch, is_best, best_score):

        state = {
            'epoch': epoch + 1,
            'state_dict': self._model.state_dict(),
            'best_prec1': best_score,
            'optimizer': optimizer.state_dict(),
            'arch': config.model
        }
        boilerplate.save_checkpoint(state, epoch, is_best, filename=self.__checkpoint_file.format(self.__cur_fold),
                                    best_filename=self.__best_model_filename.format(self.__cur_fold))

    def _split_data(self, folder, train_percent, allow_caching=True):
        random.seed(self.__seed)

        if os.path.exists(SPLIT_FILE) and allow_caching and train_percent == 0.8: # TODO: better
            #Reproducibility on several machine
            logger.info('Loading train/val split from file')

            paths = os.listdir(folder)
            cached_train = set((line.rstrip().split('.')[0] for line in open(SPLIT_FILE)))
            train_names = set([path for path in paths if path.split('.')[0] in cached_train])
            val_names = set([path for path in paths if path.split('.')[0] not in cached_train])
        else:
            train_names, val_names = set(), set()

            for filename in os.listdir(folder):
                ddict = train_names if random.random() < train_percent else val_names
                ddict.add(filename)

        return train_names, val_names

    def __get_loader(self, config):
        dt = config.data_type
        loader = dt == 'jpg' and default_loader or dt == 'tif' and tif_loader or dt == 'tif-index' and tif_index_loader \
                 or dt == 'mix' and mix_loader or dt == 'jpg_numpy' and jpg_cv2_loader \
                 or dt == 'jpg_nir' and jpg_nir_loader or None

        if loader is None:
            raise ValueError, 'No loader for the data_type=%s' % (dt, )
        return loader

    def _get_data_loader(self, config, names=None):
        transformations = self.__transform_func()
        if names is None:
            train_names, val_names = self._split_data(self.__train_dir, config.train_percent)
        else:
            train_names, val_names = names

        loader = self.__get_loader(config)

        train_folder = ImageFolder(self.__label_file, self.__train_dir, train_names, transform=transformations['train'],
                                   loader=loader)
        val_folder = ImageFolder(self.__label_file, self.__train_dir, val_names, transform=transformations['val'],
                                 loader=loader)
        if not len(train_folder) or not len(val_folder):
            raise ValueError, 'One of the image folders contains zero data, train: %s, val: %s' % \
                              (len(train_folder), len(val_folder))

        sampler = None
        if config.weigh_sample:
            sampler = WeightedRandomSampler(train_folder.weights, len(train_folder), replacement=True)

        train_loader = torch.utils.data.DataLoader(train_folder, batch_size=config.batch_size, shuffle=True,
                                                   sampler=sampler, num_workers=config.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_folder, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=config.workers, pin_memory=True)

        return train_loader, val_loader

    def _init_loss(self, weigh_loss):
        if not weigh_loss:
            # straight forward
            criterion = nn.MultiLabelSoftMarginLoss().cuda()
            logger.info('Loss is %s without weights', criterion.__class__.__name__)
        else:
            # weighing by occurence
            df = pd.read_csv(self.__label_file)
            cnts = defaultdict(int)
            for k, v in df.values:
                for label in v.split():
                    cnts[label] += 1

            total_sum = float(sum(cnts.values()))
            weights = []
            for k, v in sorted(cnts.iteritems()):
                weights.append(1-v/total_sum)

            criterion = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor(weights)).cuda()
            logger.info('Loss is %s without weights: %s', criterion.__class__.__name__, str(weights))

        return criterion

    def _init_lr_scheduler(self, config, optimizer, best_score):
        lr_scheduler = boilerplate.PlateauScheduler(optimizer, config.max_stops, config.early_stop_n,
                                                    decrease_rate=config.decrease_rate,
                                                    best_score=best_score, warm_up_epochs=config.warm_up_epochs)
        if config.lr_schedule == 'frozen':
            lr_scheduler.patience = 99999

        return lr_scheduler

    def _load_best_model(self):
        fname = self.__best_model_filename.format(self.__cur_fold)
        checkpoint = boilerplate.load_checkpoint(fname)
        self._model.load_state_dict(checkpoint['state_dict'])
        logger.info('Loaded best model %s, arch=%s', fname, checkpoint.get('arch', ''))

    def train_single_model(self, config, train_loader, val_loader):
        # variables: split, maybe some params
        logger.info('Starting learning single model')
        self._init_model()
        self.__rm_prev_checkpoints()

        optimizer = boilerplate.init_optimizer(self._model, config, exact_layers=self.__layers_to_optimize)
        start_epoch, best_score = self._init_checkpoint(optimizer, config)
        lr_scheduler = self._init_lr_scheduler(config, optimizer, best_score)

        criterion = self._init_loss(config.weigh_loss)

        for epoch in xrange(start_epoch, config.epoch):
            logger.info('Epoch %d', epoch)
            # iterate
            train_score = boilerplate.train(train_loader, self._model, criterion, optimizer, epoch)
            score = boilerplate.validate(val_loader, self._model, criterion, activation=torch.sigmoid)
            self.monitor.add_performance('loss', train_score, score)

            # change lr and save checkpoint
            adjusted, to_break, is_best = lr_scheduler.step(epoch, score)
            self._save_checkpoint(config, optimizer, epoch, is_best, best_score)
            if to_break:
                logger.info('Exiting learning process')
                break
            # load model if plateau == True
            if adjusted and config.lr_schedule == 'adaptive_best':
                logger.info('Loaded the model from previous best step')
                self._load_best_model()

    def run(self, config):
        train_loader, val_loader = self._get_data_loader(config)
        self.train_single_model(config, train_loader, val_loader)
        self.test_and_submit(config, val_loader)

    def __read_holdout(self):
        with open(HOLDOUT_FILE) as rf:
            exclude_set = set((w.rstrip('\n') for w in rf.readlines()))
        logger.info('Loaded holdout filenames, length %d', len(exclude_set))
        return exclude_set

    def __read_blacklist(self):
        with open(BLACKLIST_FILE) as rf:
            exclude_set = set((w.rstrip('\n') for w in rf.readlines()))
        logger.info('Loaded blacklisted filenames, length %d', len(exclude_set))
        return exclude_set

    def _split_kfold(self, folder, kfolds, holdout=None, blacklist=None):
        random.seed(self.__seed)
        fnames = os.listdir(folder)
        exclude_set = set()
        if holdout:
            exclude_set.update(self.__read_holdout())
        if blacklist:
            exclude_set.update(self.__read_blacklist()) # TODO: for normal training
        if exclude_set:
            logger.info('Total exclude set is %d', len(exclude_set))
            fnames = [fname for fname in fnames if fname.split('.')[0] not in exclude_set]
        fnames = np.array(fnames)

        skf = KFold(n_splits=kfolds, shuffle=True, random_state=self.__seed)
        for train_indexes, val_indexes in skf.split(fnames):
            train_names = set(fnames[train_indexes])
            val_names = set(fnames[val_indexes])
            yield train_names, val_names

    def run_ensemble(self, config):
        folds = self._split_kfold(self.__train_dir, config.folds, holdout=config.holdout, blacklist=config.blacklist)
        for i, (train_names, val_names) in enumerate(folds):
            logger.info('Fold=%d/%d', i, config.folds - 1)
            logger.info('Train/val %d/%d', len(train_names), len(val_names))
            self.__cur_fold = i
            train_loader, val_loader = self._get_data_loader(config, (train_names, val_names))
            self.train_single_model(config, train_loader, val_loader)
            thresholds = self.test_and_submit(config, val_loader)
            logger.info('Fold=%d, thresholds=%s', i, str(thresholds))
            self.monitor = boilerplate.VisdomMonitor(port=80)
            # out-of-fold prediction
            logger.info('Predicting out-of-fold (valid) dataset')
            val_folder = ImageTestFolder(self.__train_dir, transform=self.__transform_func()['test'],
                                         loader=self.__get_loader(config))
            val_folder.imgs = [fname for fname in val_folder.imgs if fname in val_names]
            names, final_results = self.test_model(config, val_folder)
            self.__write_submission(izip(names, final_results), thresholds, output_file='/dev/null',
                                    detailed_output_file=self.__valid_output_file.format(self.__cur_fold))

            # prediction for holdout
            if config.holdout == 1:
                logger.info('Predicting holdout dataset')
                # init test folder
                filenames_set = self.__read_holdout()
                test_folder = ImageTestFolder(self.__train_dir, transform=self.__transform_func()['test'],
                                              loader=self.__get_loader(config))
                test_folder.imgs = [fname for fname in test_folder.imgs if fname.split('.')[0] in filenames_set]
                # made actual predicitions
                names, final_results = self.test_model(config, test_folder)
                self.__write_submission(izip(names, final_results), thresholds, output_file='/dev/null',
                                        detailed_output_file=self.__holdout_output_file.format(self.__cur_fold))
            # end of iteration

    def estimate_thresholds(self, config, val_loader=None):
        logger.info('Estimating thresholds')
        if val_loader is None:
            logger.info('Reloading val_loader')
            _, val_loader = self._get_data_loader(config)

        # just gets raw predictions
        val_outputs, val_targets = boilerplate.get_outputs(val_loader, self._model, activation=torch.sigmoid)
        # tensor -> numpy for sk_learn
        val_outputs = np.array(map(lambda x: x.numpy(), val_outputs))
        val_targets = np.array(map(lambda x: x.numpy(), val_targets))
        label_names = val_loader.dataset.classes
        class_freq = val_loader.dataset.class_freq

        # optimize thresholds
        def optimise_f2_thresholds(y, p, steps=100):
            def get_fbeta(x):
                p2 = np.zeros_like(p)
                for i in range(NUM_CLASSES):
                    p2[:, i] = (p[:, i] > x[i]).astype(np.int)
                return fbeta_score(y, p2, beta=2, average='samples')

            best_thresholds = [0.2] * NUM_CLASSES
            for j in range(NUM_CLASSES):
                best_threshold, best_score = 0, 0
                for i2 in range(steps):
                    i2 /= float(steps)
                    best_thresholds[j] = i2
                    score = get_fbeta(best_thresholds)
                    if score > best_score:
                        best_threshold, best_score = i2, score
                best_thresholds[j] = best_threshold
                logger.info('\t slice=%d, name=%s, best_threshold=%.3f, metric=%.4f', j, label_names[j], best_threshold,
                            best_score)

            return best_thresholds

        results = optimise_f2_thresholds(val_targets, val_outputs)

        # pre/re metrics
        logger.info('Precision/recall for each category')
        for i in xrange(NUM_CLASSES):
            best_threshold = results[i]
            name = label_names[i]
            X, Y = val_outputs.T[i], val_targets.T[i]  # predictions/targets for current class
            X_new = np.vectorize(lambda x: x > best_threshold and 1 or 0)(X)
            pre = precision_score(Y, X_new)
            recall = recall_score(Y, X_new)
            logger.info('\t name=%s, count=%d, pre=%.4f, recall=%.4f', name, class_freq[name], pre, recall)

        # validate F2 measure
        f2_scores, f2_asis_scores = 0.0, 0.0
        for i, output in enumerate(val_outputs):
            target = val_targets[i]
            output_new = np.array([output[j] > results[j] and 1 or 0 for j in xrange(NUM_CLASSES)], dtype=np.float32)
            f2_scores += fbeta_score(target, output_new, 2)
            f2_asis_scores += fbeta_score(target, np.vectorize(lambda x: x > 0.5 and 1 or 0)(output), 2)
        logger.info('F2 score before applying tresholds is %.6f', f2_asis_scores/len(val_outputs))
        logger.info('F2 score after applying tresholds is %.6f', f2_scores / len(val_outputs))

        return results

    def test_model(self, config, test_folder=None):
        tr = self.__transform_func()['test']
        loader = self.__get_loader(config)
        if test_folder is None:
            test_folder = ImageTestFolder(self.__test_dir, transform=tr, loader=loader)

        # dummy
        # test_folder.imgs = test_folder.imgs[:10]

        results = []
        crop_num = len(tr.transforms[0])
        for index in xrange(crop_num):
            # iterate over tranformations
            logger.info('Testing transformation %d/%d', index + 1, crop_num)
            test_folder.transform.transforms[0].index = index
            test_loader = torch.utils.data.DataLoader(test_folder, batch_size=config.test_batch_size,
                                                      num_workers=config.workers, pin_memory=True)
            names, crop_results = boilerplate.test_model(test_loader, self._model, activation=torch.sigmoid)
            results.append(crop_results)

        final_results = [sum(map(lambda x: x[i].data.numpy(), results)) / float(crop_num) for i in
                         xrange(len(test_folder.imgs))]

        return names, final_results

    def __write_submission(self, res, thresholds, output_file=None, detailed_output_file=None):
        """Write final result into csv"""
        def sort_arg(filename):
            base, number = filename.split('_')
            number = int(number[:-4])
            return base, number

        mapping = {i:k for i, k in enumerate(ImageFolder(self.__label_file, self.__train_dir).classes)}
        logger.info(str(mapping))

        output_file = output_file or self.__output_file.format(self.__cur_fold)
        detailed_output_file = detailed_output_file or self.__detailed_output_file.format(self.__cur_fold)

        with open(output_file, 'w') as wf, open(detailed_output_file, 'w') as dwf:
            dwf.write(json.dumps({mapping[i]:v for i, v in enumerate(thresholds)}) + '\n')
            wf.write('image_name,tags\n')
            dwf.write('image_name,probs\n')
            for file_name, probs in sorted(res, key=lambda x: sort_arg(x[0])):
                name = file_name[:-4]
                labels, detailed_enc = [], []
                for i, prob in enumerate(probs):
                    if prob > thresholds[i]:
                        labels.append(mapping[i])
                    detailed_enc.append('%s:%.5f' % (mapping[i], prob))
                wf.write(','.join([name, ' '.join(labels)]) + '\n')
                dwf.write(','.join([name, ' '.join(detailed_enc)]) + '\n')

    def test_and_submit(self, config, val_loader=None):
        self._init_model()
        self._load_best_model()
        thresholds = self.estimate_thresholds(config, val_loader)
        names, final_results = self.test_model(config)
        self.__write_submission(izip(names, final_results), thresholds)
        return thresholds


def main():
    parser = OptionParser()
    parser.add_option("-b", "--batch_size", action='store', type='int', dest='batch_size', default=128)
    parser.add_option("--test_batch_size", action='store', type='int', dest='test_batch_size', default=320)
    parser.add_option("-e", "--epoch", action='store', type='int', dest='epoch', default=80)
    parser.add_option("-r", "--workers", action='store', type='int', dest='workers', default=2)
    parser.add_option("-l", "--learning-rate", action='store', type='float', dest='lr', default=0.01)
    parser.add_option("-m", "--momentum", action='store', type='float', dest='momentum', default=0.9)
    parser.add_option("-w", "--weight_decay", action='store', type='float', dest='weight_decay', default=1e-4)
    parser.add_option("-o", "--optimizer", action='store', type='string', dest='optimizer', default='sgd',
                      help='sgd|adam|yf')
    parser.add_option("-c", "--from_checkpoint", action='store', type='int', dest='from_checkpoint', default=0,
                      help='resums training from a specific epoch')
    parser.add_option("--train_percent", action='store', type='float', dest='train_percent', default=0.8,
                      help='train/val split percantage')
    parser.add_option("--decrease_rate", action='store', type='float', dest='decrease_rate', default=0.1,
                      help='For lr schedule, on plateau how much to descrease lr')
    parser.add_option("--early_stop_n", action='store', type='int', dest='early_stop_n', default=6,
                      help='Early stopping on a specific number of degrading epochs')
    parser.add_option("--folds", action='store', type='int', dest='folds', default=10,
                      help='Number of folds, for ensemble training only')
    parser.add_option("--lr_schedule", action='store', type='str', dest='lr_schedule', default='adaptive_best',
                      help="""possible: adaptive, adaptive_best, decreasing or frozen.
                       adaptive_best is the same as plateau scheduler""")
    parser.add_option("--model", action='store', type='str', dest='model', default=None,
                      help='Which model to use, check model.py for names')
    parser.add_option("--transform", action='store', type='str', dest='transform', default=None,
                      help='Specify a transformation rule. Check transform_rules.py for names')
    parser.add_option("--weigh_loss", action='store', type='int', dest='weigh_loss', default=0,
                      help='weigh loss function according to class occurence or not')
    parser.add_option("--warm_up_epochs", action='store', type='int', dest='warm_up_epochs', default=2,
                      help='warm_up_epochs number if model has it')
    parser.add_option("--max_stops", action='store', type='int', dest='max_stops', default=2,
                      help='max_stops for plateau/adaptive lr schedule')
    parser.add_option("--weigh_sample", action='store', type='int', dest='weigh_sample', default=0,
                      help='weigh sample according to class occurrence or not')
    parser.add_option("--data_type", action='store', type='str', dest='data_type', default='jpg',
                      help='Data type. Possible values: jpg|tif|tif-index|mix|jpg_numpy')
    parser.add_option("--run_type", action='store', type='str', dest='run_type', default='train',
                      help='train|eval')
    parser.add_option("--seed", action='store', type='int', dest='seed', default=0)
    parser.add_option("--shard", action='store', type='str', dest='shard', default='',
                      help='Postfix for results folder, where the results will be saved, <results+shard>/')
    parser.add_option("--holdout", action='store', type='int', dest='holdout', default=0,
                      help='if eq. 1, then small 10% holdout set is not used for training, but for blending later')
    parser.add_option("--blacklist", action='store', type='int', dest='blacklist', default=0,
                      help='Use blacklist file of garbage images or labels, *Used for ensembles only* ')

    # Options
    config, _ = parser.parse_args()
    assert config.lr_schedule in ('adaptive', 'decreasing', 'frozen', 'adaptive_best')
    assert config.model is not None
    try:
        model_func = getattr(model, config.model)
    except AttributeError:
        raise AttributeError, "Model %s doesn't exist" % config.model
    try:
        transform_func = getattr(transform_rules, config.transform)
    except AttributeError:
        raise AttributeError, "Transform %s doesn't exist" % config.transform

    # Init trainer
    from __init__ import TRAIN_FOLDER_JPG, TEST_FOLDER_JPG, LABEL_FILE, TRAIN_FOLDER_TIF, TEST_FOLDER_TIF
    if config.data_type in ('jpg', 'jpg_numpy'):
        train_folder = TRAIN_FOLDER_JPG
        test_folder = TEST_FOLDER_JPG
        pre_trained = True
    elif config.data_type in ('mix', 'jpg_nir'):
        train_folder = TRAIN_FOLDER_TIF
        test_folder = TEST_FOLDER_TIF
        pre_trained = True
    else:
        train_folder = TRAIN_FOLDER_TIF
        test_folder = TEST_FOLDER_TIF
        pre_trained = False

    tr = Trainer(model_func, transform_func, LABEL_FILE, train_folder, test_folder, pre_trained=pre_trained,
                 seed=config.seed, shard=config.shard)
    logger.info('Config: %s', str(config))
    # Run
    try:
        if config.run_type == 'train':
            tr.run(config)
        elif config.run_type == 'eval':
            tr.test_and_submit(config)
        elif config.run_type == 'ens':
            tr.run_ensemble(config)
    except:
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    sys.exit(main())
