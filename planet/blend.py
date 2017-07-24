#-*- coding: utf8 -*-
import os, sys, glob, json, numpy as np, copy
from optparse import OptionParser
from scipy.stats.mstats import gmean
from sklearn.metrics import log_loss, fbeta_score
from __init__ import LABEL_FILE, TRAIN_FOLDER_TIF, NUM_CLASSES
from folder import ImageFolder
import torch, torch.nn as nn
from torch.nn import functional as F
import operator


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def gather_directories(folder):
    tests, trains = [], []
    splits = []
    for _dir in sorted(os.listdir(folder)):
        test = glob.glob(os.path.join(folder, _dir, 'detailed_submission*.txt'))
        train = glob.glob(os.path.join(folder, _dir, 'detailed_holdout*.txt'))
        test.sort()
        train.sort()
        tests.extend(test)
        trains.extend(train)
        splits.append(len(test))

    return trains, tests, splits


def read_file(fname):
    print 'Reading file %s' % fname
    predictions = {}

    with open(fname) as rf:
        thresholds = json.loads(rf.readline())
        thresholds = np.array([v for k,v in sorted(thresholds.iteritems())])
        print thresholds
        rf.readline()
        for line in rf:
            name, labels = line.split(',')
            predictions[name] = np.array([float(w.split(':')[1]) for w in labels.split(' ')])
    return predictions, thresholds


def get_groundtruth(permitted):
    folder = ImageFolder(LABEL_FILE, TRAIN_FOLDER_TIF, set(permitted))
    truth = map(lambda x: (x[0].split('.')[0], x[1].numpy()), folder.imgs)
    truth.sort()

    names, data = zip(*truth)
    data = np.array(data)

    return data

def thr_pred(p, thrs):
    p2 = np.zeros_like(p)
    for i in range(NUM_CLASSES):
        p2[:, i] = (p[:, i] > thrs[i]).astype(np.int)
    return p2


def get_fbeta_x(y, p, thrs): # TODO: trash
    p2 = thr_pred(p, thrs)
    return fbeta_score(y, p2, beta=2, average='samples')



def main():
    parser = OptionParser()
    parser.add_option("--folder", action='store', type='str', dest='folder', default=None,
                      help="""path to folder with others folders containing predictions from first stage models.
                       First stage models must content holdout predictions""")
    config, _ = parser.parse_args()
    folder = config.folder
    assert folder is not None and os.path.exists(folder)

    nss = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine',
     'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn',
     'water']
    print 'Folder', folder
    trains, tests, splits = gather_directories(folder)
    print 'Number of files %d' % len(trains)
    print trains
    g_preds, g_thrs = zip(*(read_file(fname) for fname in trains))
    avg_thr = gmean(g_thrs)
    print 'avg_thr', avg_thr
    keys = g_preds[0].keys()
    keys.sort()
    Y = get_groundtruth(map(lambda x: x + '.tif', keys))
    Xss = [np.array([v for _, v in sorted(gpred.iteritems())]) for gpred in g_preds]
    Xs = Xss
    #Xs = map(lambda z: np.sum(z, axis=0)/float(len(z)), np.split(Xss, list(accumulate(splits))[:-1]))


    X = np.array(Xs).transpose(1,0,2)
    X_train, X_valid = X[:400], X[400:]
    Y_train, Y_valid = Y[:400], Y[400:]

    channels = len(Xs)
    model = nn.Sequential(
        #nn.Conv1d(channels, channels, 1, stride=1, bias=False),
        #nn.ELU(inplace=True),
        #nn.BatchNorm1d(channels),
        nn.Conv1d(channels, 1, 1, stride=1, bias=True),
        #nn.Conv1d(channels, 17, 17, stride=17, bias=True),
        #nn.Sigmoid(),
    )
    model.apply(weights_init)

    X_train = torch.from_numpy(X_train.astype('float32'))
    X_valid = torch.from_numpy(X_valid.astype('float32'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    criterion = nn.MultiLabelSoftMarginLoss()
    best_score, best_model_dict = 100, {}
    patience, max_patience = 0, 20
    print 'conv OFF'
    for epoch in xrange(1):
        input_var = torch.autograd.Variable(X_train)
        target_var = torch.autograd.Variable(torch.from_numpy(Y_train))
        output = model(input_var)
        loss = criterion(output.squeeze(), target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.data[0]

        #val
        model.eval()
        input_val_var = torch.autograd.Variable(X_valid, volatile=True)
        target_val__var = torch.autograd.Variable(torch.from_numpy(Y_valid), volatile=True)
        output = model(input_val_var)
        loss_val = criterion(output.squeeze(), target_val__var)
        loss_data_value = loss_val.data[0]
        model.train()


        if loss_data_value + 1e-4 < best_score:
            best_score = loss_data_value
            best_model_dict = copy.deepcopy(model.state_dict())
            patience = 0
            #print list(model.parameters())[0].data.numpy()
        else:
            patience += 1
        print 'Epoch', epoch, 'loss', loss_value * 100, loss_data_value * 100, patience
        if patience == max_patience:
            print 'Early stop'
            break
    print 'The end of training blend'
    # Test time
    print 'Loading best model'
    model.load_state_dict(best_model_dict)
    model.eval()
    #print list(model.parameters())
    weights = list(model.parameters())[0].data.numpy().squeeze()
    weights = (weights - np.min(weights))/np.max(weights)
    weights = weights / np.sum(weights)
    print 'weights', weights

    # current losses
    print 'Calculating loss without fucking classifiers, just on f2-0.5 loss'
    new_weights = np.zeros(len(weights))
    for i, _X in enumerate(Xs):
        half = get_fbeta_x(Y, _X, [0.5] * NUM_CLASSES)
        optimum = get_fbeta_x(Y, _X, avg_thr)
        print trains[i], 'weight_tr=', weights[i], log_loss(Y, _X), optimum , half
        new_weights[i] = half
    new_weights = (new_weights - np.min(new_weights)) / np.max(new_weights)
    new_weights = new_weights ** 0.1
    new_weights = new_weights / np.sum(new_weights)

    weights = new_weights
    print 'new_weights', weights
    print 'weight sum', np.sum(weights)

    first_test, _ = zip(*(read_file(fname) for fname in tests))
    test_keys = first_test[0].keys()
    test_keys.sort()
    test_fulls = [np.array([v for _, v in sorted(tpred.iteritems())]) for tpred in first_test]
    #test_arrays = map(lambda z: np.sum(z, axis=0) / float(len(z)), np.split(test_fulls, list(accumulate(splits))[:-1]))
    test_arrays = test_fulls
    X_test = np.array(test_arrays).transpose(1, 0, 2)

    #
    # test_var = torch.autograd.Variable(torch.from_numpy(X_test.astype('float32')), volatile=True)
    # output = F.sigmoid(model(test_var)).squeeze().data
    #
    # for i in xrange(len(output)):
    #     print test_keys[i]
    #     print (X_test[i]*100).astype('int')
    #     print (output[i].numpy() * 100).astype('int')
    #     #print test_keys[i], zip(nss, (output[i].numpy() * 100).astype('int'))
    #
    #     new_output = np.sum([w * weights[j] for j, w in enumerate(X_test[i])], axis=0)
    #     print (new_output * 100).astype('int')
    #
    #     if i > 5:
    #         break

    print 'gmean', avg_thr

    after_X = np.zeros(shape=(X.shape[0], X.shape[2]))
    print after_X.shape
    for i in xrange(len(after_X)):
        after_X[i] = np.sum([w * weights[j] for j, w in enumerate(X[i])], axis=0)
    print '\nfinal scores\n', log_loss(Y, after_X), get_fbeta_x(Y, after_X, avg_thr), \
            get_fbeta_x(Y, after_X, [0.5]*NUM_CLASSES)

    print 'avg thr', avg_thr
    out_file = '/home/tyantov/workspace/kaggle-planet/results/avg_subm_blend.csv'
    with open(out_file, 'w') as wf:
        wf.write('image_name,tags\n')
        for i in xrange(len(X_test)):
            name = test_keys[i]
            res = []
            out = np.sum([w * weights[j] for j, w in enumerate(X_test[i])], axis=0)
            for j, prob in enumerate(out):
                if prob > avg_thr[j]:
                    res.append(nss[j])
            wf.write('%s,%s\n' % (name, ' '.join(res)))
    print 'Done'

if __name__ == '__main__':
    sys.exit(main())
