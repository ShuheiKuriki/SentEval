# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TREC question-type classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier


class TRECEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train_5500.label'))
        self.test = self.loadFile(os.path.join(task_path, 'TREC_10.label'))
        self.classnum = 6

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}
        tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
                   'HUM': 3, 'LOC': 4, 'NUM': 5}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1].split()
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data

    def set_data(self, params, vectorizer):
        trainset, testset = [], []

        # Sort to reduce padding
        corpus_train = list(zip(self.train['X'], self.train['y']))

        train_sentences = [x for (x, y) in corpus_train]
        trainsetlabel = [y for (x, y) in corpus_train]

        # Get train embeddings
        train_embeddings = vectorizer(params, train_sentences)

        trainset = [[] for i in range(self.classnum)]
        trainsetnum = [0]*self.classnum
        for emb, label in zip(train_embeddings, trainsetlabel):
            trainset[label].append(emb)
            trainsetnum[label] += 1

        corpus_test = list(zip(self.test['X'], self.test['y']))
        test_sentences = [x for (x, y) in corpus_test]
        testsetlabel = [y for (x, y) in corpus_test]

        test_embeddings = vectorizer(params, test_sentences)

        testset = [[] for i in range(self.classnum)]
        testsetdata = []
        testsetnum = [0]*self.classnum
        for emb, label in zip(test_embeddings, testsetlabel):
            testset[label].append(emb)
            testsetdata.append(emb)
            testsetnum[label] += 1
        testsetdatanum = sum(testsetnum)

        

        # config_classifier = {'nclasses': 6, 'seed': self.seed,
        #                      'usepytorch': params.usepytorch,
        #                      'classifier': params.classifier,
        #                      'kfold': params.kfold}
        # clf = KFoldClassifier({'X': trainset,
        #                        'y': np.array(trainsetlabel)},
        #                       {'X': testset,
        #                        'y': np.array(testsetlabel)},
        #                       config_classifier)
        # devacc, testacc, _ = clf.run()
        # logging.debug('\nDev acc : {0} Test acc : {1} \
        #     for TREC\n'.format(devacc, testacc))
        # return {'devacc': devacc, 'acc': testacc,
        #         'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}

        return trainset, trainsetnum, testset, testsetdata, testsetdatanum, testsetlabel, testsetnum

