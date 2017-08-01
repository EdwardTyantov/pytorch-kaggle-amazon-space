#-*- coding: utf8 -*-
import os, sys, glob, json, numpy as np
from optparse import OptionParser
from folder import ImageFolder
from collections import defaultdict
from scipy.stats.mstats import gmean
from sklearn.metrics import log_loss, fbeta_score
from __init__ import LABEL_FILE, TRAIN_FOLDER_TIF, NUM_CLASSES, BLACKLIST_FILE
from folder import ImageFolder
import torch, torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from stacked_generalization.lib.stacking import StackedClassifier


with open(BLACKLIST_FILE) as rf:
    blacked_labels = set((w.rstrip('\n') for w in rf.readlines()))

def posmax(seq, key=lambda x: x):
    return max(enumerate(seq), key=lambda k: key(k[1]))[0]

# perm = """5_selu_hv 5_mixnet6_hvb 6_mixnet6_hvb  7_mixnet6_hvb  7_mixnetv3_hvb  8_mixnet6_hvb 8_mixnetv3_hvb"""
# perm = [w for w in perm.split() if w]

def gather_directories(folder):
    print 'TESTING'
    dirs, tests, trains, valids = [], [], [], []
    for _dir in sorted(os.listdir(folder)):
        # if _dir not in perm:
        #    print 'SKIP', _dir
        #    continue
        train = glob.glob(os.path.join(folder, _dir, 'detailed_valid*.txt'))
        test = glob.glob(os.path.join(folder, _dir, 'detailed_submission*.txt'))
        valid = glob.glob(os.path.join(folder, _dir, 'detailed_holdout*.txt'))
        test.sort()
        train.sort()
        valid.sort()
        tests.append(test)
        trains.append(train)
        valids.append(valid)
        dirs.append(_dir)
    return dirs, trains, valids, tests


def read_file(fname, is_test=False):
    print 'Reading file %s' % fname
    predictions = {}

    with open(fname) as rf:
        thresholds = json.loads(rf.readline())
        thresholds = np.array([v for k,v in sorted(thresholds.iteritems())])
        #print thresholds
        rf.readline()
        for i, line in enumerate(rf):
            name, labels = line.split(',')
            predictions[name] = np.array([float(w.split(':')[1]) for w in labels.split(' ')])
            if is_test and i > 1000:
                print 'break'
                break
    return predictions, thresholds


def get_groundtruth(permitted=None):
    folder = ImageFolder(LABEL_FILE, TRAIN_FOLDER_TIF, set(permitted))
    truth = map(lambda x: (x[0].split('.')[0], x[1].numpy()), folder.imgs)
    truth.sort()

    names, data = zip(*truth)
    data = np.array(data)

    return data


def gather_train(train_files, is_test=False, blacked=True):
    out = {}
    for tfile in train_files:
        pred, _ = read_file(tfile, is_test=is_test)
        out.update(pred)
    if blacked:
        print 'Blacking out'
        for k in blacked_labels:
            if k in out:
                del out[k]
    if 'train_39193' in out:
        del out['train_39193']
    return out


def gather_test(train_files, is_test=False):
    N = float(len(train_files))
    print 'is_test', is_test
    out = defaultdict(float)
    for tfile in train_files:
        pred, _ = read_file(tfile, is_test=is_test)
        for k,v in pred.iteritems():
            out[k] += v/N
    return out


def gather_valid(train_files, is_test=False):
    N = float(len(train_files))
    out = defaultdict(float)
    thrs = []
    for tfile in train_files:
        pred, thr = read_file(tfile, is_test=is_test)
        thrs.append(thr)
        for k,v in pred.iteritems():
            out[k] += v/N
    return out, thrs


def optimise_f2_thresholds(y, p):
    thresholds = [w * 0.001 for w in xrange(100, 400, 5)]
    metrics = []
    for threshold in thresholds:
        metric = fbeta_score(y, (p > threshold).astype(np.int), 2)
        metrics.append(metric)
        #print '---', threshold, metric
    best_threshold = thresholds[posmax(metrics)]
    return best_threshold


def thr_pred(p, thrs):
    p2 = np.zeros_like(p)
    for i in range(NUM_CLASSES):
        p2[:, i] = (p[:, i] > thrs[i]).astype(np.int)
    return p2


def get_fbeta_x(y, p, thrs): # TODO: trash
    p2 = thr_pred(p, thrs)
    return fbeta_score(y, p2, beta=2, average='samples')


def overall_optimise_f2_thresholds(y, p, nss, steps=100, eps=1e-4):
    def get_fbeta(x):
        p2 = np.zeros_like(p)
        for i in range(NUM_CLASSES):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        return fbeta_score(y, p2, beta=2, average='samples')

    best_thresholds = [0.2] * NUM_CLASSES
    for j in range(NUM_CLASSES):
        best_threshold, best_score = 0, 0
        scores = []
        for i2 in range(0, 100):
            i2 /= float(steps)
            best_thresholds[j] = i2
            score = get_fbeta(best_thresholds)
            if score - eps > best_score:
                best_threshold, best_score = i2, score
            scores.append((i2, score))
        best_thresholds[j] = best_threshold
        print '\t slice=%d, name=%s, best_threshold=%.3f, metric=%.4f' % (j, nss[j], best_threshold,
                    best_score), scores

    return best_thresholds


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
    dirs, trains, valids, tests = gather_directories(folder)
    print 'Number of first-tier classifier %d' % len(trains)

    train_predictions = map(gather_train, trains)
    Xs = [np.array([v for _, v in sorted(gpred.iteritems())]) for gpred in train_predictions]
    print map(np.shape, Xs)

    X = np.array(Xs).transpose(1,0,2)
    Y = get_groundtruth(map(lambda x: x + '.tif', train_predictions[0].keys()))

    preV, pre_thrs = zip(*map(lambda  x: gather_valid(x, is_test=False), valids))
    thrs = []
    for temp in pre_thrs:
        thrs.extend(temp)
    valid_thrs = gmean(thrs)
    print 'valid_thrs', valid_thrs

    V = [np.array([v for _, v in sorted(tpred.iteritems())]) for tpred in preV]
    V = np.array(V).transpose(1, 0, 2)
    Y_v = get_groundtruth(map(lambda x: x + '.tif', preV[0].keys()))

    print 'Valid'
    # TODO: validation
    # np.array([np.array([int(v > valid_thrs[j]) for j, v in enumerate(np.mean(V[i], axis=0))]) for i in len(V)])
    print V.shape, Y_v.shape

    for idx in xrange(len(preV)):
        elem = V[:,idx,:]
        huem = (elem > 0.5).astype('int')
        for class_idx in xrange(NUM_CLASSES):
            elem[:, class_idx] = (elem[:, class_idx] > valid_thrs[class_idx]).astype('int')
        print 'ens average prediction',  os.path.basename(dirs[idx]), fbeta_score(Y_v, elem, 2, average='samples'), \
            fbeta_score(Y_v, huem, 2, average='samples')

    first_test = map(lambda  x: gather_test(x, is_test=False), tests) # is_test for fast testing this script

    test_keys = first_test[0].keys()
    test_keys.sort()
    T = [np.array([v for _, v in sorted(tpred.iteritems())]) for tpred in first_test]
    T = np.array(T).transpose(1,0,2)
    R = np.zeros((17, len(T)))
    RV = np.zeros((17, len(V)))

    opt_thresholds = []
    for class_idx in xrange(NUM_CLASSES):
        print 'Learning class=%d', class_idx
        X_class = X[:, :,class_idx]
        Y_class = Y[:, class_idx]
        T_class = T[:, :, class_idx]
        V_class = V[:, :, class_idx]
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)

        class_thresholds = []
        test_results, valid_results = [], []
        for t_idx, v_idx in skf.split(X_class, Y_class):
            X_train, X_valid = X_class[t_idx], X_class[v_idx]
            Y_train, Y_valid = Y_class[t_idx], Y_class[v_idx]

            # lr = KNeighborsClassifier(n_neighbors=5)
            # lr = RandomForestClassifier(n_estimators=25, criterion = 'gini', random_state=1)
            # lr = GradientBoostingClassifier(n_estimators=25)
            # sg = GradientBoostingClassifier(n_estimators=50, subsample=0.7)
            sg = LogisticRegression(C=100, max_iter=1000, fit_intercept=True,
                                    class_weight={0:1, 1:8})

            # base_models = [RandomForestClassifier(n_estimators=20, criterion = 'gini', random_state=1),
            #                ExtraTreesClassifier(n_estimators=20, criterion = 'gini', random_state=1),
            #                GradientBoostingClassifier(n_estimators=25, random_state=1),
            #                KNeighborsClassifier(n_neighbors=5)
            #                ]
            #
            # blending_model = LogisticRegression(C=10, fit_intercept=True, max_iter=1000,
            #                         class_weight={0:1, 1:8}) #GradientBoostingClassifier()
            #
            # sg = StackedClassifier(blending_model, base_models, n_folds=6, verbose=0)

            sg.fit(X_train, Y_train)
            res = sg.predict_proba(X_valid)[:, 1]
            #print 'out', (res[:100]*100).astype('int')
            #print 'real', (X_valid[:100, 0] * 100).astype('int')
            test_results.append(sg.predict_proba(T_class)[:, 1])
            valid_results.append(sg.predict_proba(V_class)[:, 1])

            #print res[:100, 1]
            #print Y_valid[:100]
            print fbeta_score(Y_valid, (res > 0.5).astype(np.int), 2)
            fscores = [fbeta_score(Y_valid, (X_valid[:, iw] > 0.5).astype(np.int), 2) for iw in xrange(X_valid.shape[1])]
            print 'max', max(fscores), fscores, '\n---'
            print class_idx, 'params', sg.coef_
            class_thresholds.append(optimise_f2_thresholds(Y_valid, res))

        print 'Finishing class'
        RV[class_idx] = np.mean(np.array(valid_results), axis=0)
        R[class_idx] = np.mean(np.array(test_results), axis=0)
        opt_thr = gmean(class_thresholds)
        #test = np.mean(np.array(test_results), axis=0)
        print 'Opt, thr', opt_thr, class_thresholds
        opt_thresholds.append(opt_thr)
        #class_result = (test > opt_thr).astype('int')
        #R[class_idx] = class_result
        #thrs = optimise_f2_thresholds(Y_valid, res[:, 1], nss)
        #print thrs
    R = R.transpose(1, 0)
    RV = RV.transpose(1, 0)

    print fbeta_score(Y_v, (RV > 0.5).astype('int'), 2, average='samples')
    #thrs = overall_optimise_f2_thresholds(Y_v, RV, nss)
    #print 'overall thresholds', thrs
    print 'opt_thresholds', opt_thresholds
    #print 'valid overall score', get_fbeta_x(Y_v, RV, thrs)
    print 'valid opt score', get_fbeta_x(Y_v, RV, opt_thresholds)
    print 'valid def score', get_fbeta_x(Y_v, RV, [0.5]*17)

    print 'thresholding result'
    def outputx(r, thrs, out_file):
        R = thr_pred(r, thrs)
        print 'Output'
        with open(out_file, 'w') as wf:
            wf.write('image_name,tags\n')
            for i in xrange(len(T)):
                res = []
                for j, prob in enumerate(R[i]):
                    if prob:
                        res.append(nss[j])
                wf.write('%s,%s\n' % (test_keys[i], ' '.join(res)))

    #outputx(R, thrs, '/home/tyantov/workspace/kaggle-planet/results/avg_subm_stack1.csv')
    outputx(R, opt_thresholds, '/home/tyantov/workspace/kaggle-planet/results/avg_subm_stack2.csv')
    outputx(R, [0.5]*17, '/home/tyantov/workspace/kaggle-planet/results/avg_subm_stack3.csv')
    sys.exit()






if __name__ == '__main__':
    sys.exit(main())
