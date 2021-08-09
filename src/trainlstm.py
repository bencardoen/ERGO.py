# Copyright (C) 2018-2021  Ben Cardoen
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as published
#     by the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from loader import LoadData, ToTensor, SVRGDataset, findframe, findframerange, evaluate, findpredicted
from model import LSTMSVRG, makeNet, evaluateModel
from plotter import quantifyLoss, quantifyDistribution, quantifyPrediction, quantifyConfusion
import copy
from tqdm import tqdm
from copy import deepcopy
import torch
import os
import numpy as np
import seaborn as sns
from datetime import datetime
import argparse
import json
from gconf import initlogger, getlogger
from sklearn.utils import class_weight
lgr = None


configuration = {'epochs':10,'loss':'mse', 'batch':64, 'lr':0.00001,
                 'imageroot': "../data",
                 'csv': "../data/density.csv",
                 'outputroot': "/dev/shm", 'pix':15, "batchweights":False, 'model':'lstm', 'shuffle':False,
                 'mode':'train', 'pretraineddir':None, 'trimclasses':5, 'weightstrategy':'none', 'seed':1,
                 'initstrategy':'zero'}


def storeconfig(conf):
    lgr.info("Storing configuration ...")
    safe = copy.copy(conf)
    del safe['device']
    with open(os.path.join(conf['outputdirectory'], 'config.json'), 'w') as fp:
        json.dump(safe, fp)
    lgr.info("... Done")


def evaldata(dataset, defaultvalues):
    ax = sns.distplot(np.array(dataset.gt_frame['count']), kde=False)
    ax.set(xlabel='Count', ylabel='Frequency')
    ax.set_title('Distribution of counts in total dataset.')
    ax.get_figure().savefig(os.path.join(defaultvalues['outputdirectory'], 'dataset_count_distribution.png'))


def initialize():
    global lgr
    torch.manual_seed(1)
    acceptablestrategies=['none', 'batch', 'global', 'sklearn']
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--epochs', type=int, nargs='?', help='Int, >0, Epochs')
    parser.add_argument('--seed', type=int, nargs='?', help='Seed, default 1')
    parser.add_argument('--batch', type=int, nargs='?', help='Int, >0, Batch size, (default 64')
    parser.add_argument('--batchweights', type=str, nargs='?', help='Default off If true compute weights per batch to offset imbalance')
    parser.add_argument('--weightstrategy', type=str, nargs='?',help='Default : none, options: none, global, sklearn, batch (if --batchweights=True')
    parser.add_argument('--lr', type=float, nargs='?', help='Float, >0, Learning Rate')
    parser.add_argument('--imageroot', type=str, nargs='?', help='Directory containing rois')
    parser.add_argument('--csv', type=str, nargs='?', help='Ground truth csv file location')
    parser.add_argument('--outputroot', type=str, nargs='?',help='Output root directory, will be created if not exists')
    parser.add_argument('--loss', type=str, nargs='?',help='mse, intmse, exp, intexp')
    parser.add_argument('--model', type=str, nargs='?', help='model type: fcn, lstm')
    parser.add_argument('--pix', type=int, nargs='?', help='Pixel size of ROI')
    parser.add_argument('--shuffle', type=str, nargs='?', help='Default False, decides if the training data is shuffled')
    parser.add_argument('--logdir', type=str, nargs='?', help='logging directory')
    parser.add_argument('--mode', type=str, nargs='?', help='train, test, validate, evaluate')
    parser.add_argument('--initstrategy', type=str, nargs='?', help='zero, random')
    parser.add_argument('--pretraineddir', type=str, nargs='?', help='directory where the corresponding pth file can be found, only applied if mode==evaluate')
    parser.add_argument('--trimclasses', type=int, nargs='?', help='Limit the number of classes. Useful to discount effect of <1% outliers.')
    args = parser.parse_args()
    overrides = []
    for k in configuration:
        try:
            argk = getattr(args, k)
            if argk is not None:
                overrides.append("Overriding {} : {} -> {}".format(k, configuration[k], argk))
                if k == 'weightstrategy':
                    if argk not in acceptablestrategies:
                        print("Invalid weight strategy {}".format(argk))
                        raise AssertionError('Invalid weight strategy! {}'.format(argk))
                if k == 'shuffle':
                    argk = (argk== 'True')
                if k == 'batchweights':
                    argk = (argk == 'True')
                configuration[k] = argk
        except AttributeError as e:
            continue
    OUTPUTROOT = configuration['outputroot']
    outputdirectory = os.path.join(OUTPUTROOT, "{}_{}_{}".format(datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss"), str(np.random.randint(1000)), configuration['seed']))
    assert (not os.path.exists(outputdirectory))
    os.makedirs(outputdirectory)
    os.path.exists(outputdirectory)
    configuration['outputdirectory'] = outputdirectory
    configuration['logdir'] = outputdirectory
    lgr = initlogger(configuration)
    lgr.info("Writing output in {}".format(outputdirectory))
    configuration['outputdirectory'] = outputdirectory

    if configuration['batchweights']:
        configuration['weightstrategy'] = 'batch'
    if configuration['weightstrategy'] == 'batch':
        configuration['batchweights'] = True
    lgr.info("Weight strategy {} {}".format(configuration['weightstrategy'], configuration['batchweights']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lgr.info("Running in mode : {}".format(configuration['mode']))
    lgr.info("Running on {}".format(device))
    PIX = configuration['pix']
    IMAGEROOT = configuration['imageroot']
    lgr.info("Image root is {}".format(IMAGEROOT))
    CSV = configuration['csv']
    assert (os.path.exists(IMAGEROOT))
    assert (os.path.exists(CSV))
    lgr.info("Parameter overrides:")
    for o in overrides:
        lgr.info(o)
    assert (os.path.exists(IMAGEROOT))
    assert (os.path.exists(CSV))
    lgr.debug("CONF::\t Using configuration :: ")
    for k, v in configuration.items():
        lgr.debug("CONF::\t\t {} -> {}".format(k, v))

    torch.manual_seed(configuration['seed'])
    configuration['device'] = device


def computeWeights(dataset, config):
    t = [l / sum(dataset.ftable) for l in dataset.ftable]
    classes = len(t)
    weights = None
    lgr.info("Weight strategy is {}".format(configuration['weightstrategy']))
    if config['weightstrategy'] == 'none':
        weights = np.array([1.0 / classes for i in range(classes)])
    elif config['weightstrategy'] == 'global':
        weights = np.array([2 - _t for _t in t])
    elif config['weightstrategy'] == 'sklearn':
        ys = np.array(dataset.gt_frame['count'])
        weights = class_weight.compute_class_weight('balanced', [i for i in range(classes)], ys)
    elif config['weightstrategy'] == 'batch':
        weights = np.array([1.0 / classes for i in range(classes)])
    else:
        lgr.error("Invalid weight strategy")
        raise(AssertionError('Invalid weight strategy.'))
    lgr.info("Weights {}".format(weights))
    tensor_weights = torch.from_numpy(weights).type(torch.FloatTensor).cuda()
    config['weights'] = weights.tolist()
    return tensor_weights


def prepareData(dataset, configuration):

    BATCHSIZE=configuration['batch']
    N = len(dataset)
    sp = N / 10 * 7.2 # Range of training
    vp = N / 10 * 9 # Range of 0-validation
    # Barrier ensures no frame correlation is possible between train/test/validate
    barrier = 60
    tr = int(sp)
    Tr = findframe(tr, dataset.mapping)
    Tb = findframe(Tr + barrier, dataset.mapping)
    te = int(vp)
    Te = findframe(te, dataset.mapping)
    Tv = findframe(Te + barrier, dataset.mapping)
    # Testset is from Tv -> End
    lgr.info("Validation set  rois = {} {}".format(Tv, N))
    lgr.info(findframerange(Tv, N - 1, dataset.mapping))


    teframebegin, teframeend = findframerange(Tb, Te, dataset.mapping)
    lgr.info("Test set frame range : {} - {}".format(teframebegin, teframeend))
    trframebegin, trframeend = findframerange(0, tr, dataset.mapping)
    lgr.info("Train set frame range : {} - {}".format(trframebegin, trframeend))

    # Training
    shuffle = configuration['shuffle']
    lgr.info("Shuffling Training Data ? {}".format(shuffle))

    trvrg = None
    if configuration['model'] == 'lstm':
        trvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor())
    elif configuration['model'] == 'fcn':
        trvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor(),augment='balanced')
    else:
        lgr.error("No supported model  in configuration ...")
        raise ValueError('Invalid model in configuration')
    # Augmented
    # LSTM only
    t9rvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor(),
                         augment='90')
    t8rvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor(),
                         augment='180')
    t7rvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor(),
                         augment='270')
    nvrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tr)], transform=ToTensor(),
                       augment='noise')
    # Test
    tevrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tb, Te)], transform=ToTensor())
    # Validation
    vavrg = SVRGDataset(data=deepcopy(dataset.data), indices=[i for i in range(Tv, N)], transform=ToTensor())

    # Create the loaders
    trframes = trframeend - trframebegin
    trrois = len(trvrg)
    lgr.info("Training set has {} ROIs split over {} frames is {:.2f}".format(trrois, trframes, trrois / trframes))

    #change Loader for FCN
    strainloader = torch.utils.data.DataLoader(trvrg, batch_size=BATCHSIZE, shuffle=configuration['shuffle'], num_workers=0)
    # Todo : refactor & cleanup
    nloader = torch.utils.data.DataLoader(nvrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    aug9loader = torch.utils.data.DataLoader(t9rvrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    aug8loader = torch.utils.data.DataLoader(t8rvrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    aug7loader = torch.utils.data.DataLoader(t7rvrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    stestloader = torch.utils.data.DataLoader(tevrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    svalidloader = torch.utils.data.DataLoader(vavrg, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    lgr.info("Training {0:.2f} % Testing {1:.2f} % Validation {2:.2f} %".format(Tr / N * 100, (Te - Tb) / N * 100,
                                                                                (N - Tv) / N * 100))
    #change for LSTM FCN
    trainloaders = [strainloader, aug9loader, aug8loader, aug7loader, nloader] if configuration['model'] == 'lstm' else [strainloader]
    testloaders = [stestloader]
    validloaders = [svalidloader]
    lgr.info("Size of training is {} batches".format(len(strainloader)))
    lgr.info("Size of testing is {} batches".format(len(stestloader)))
    lgr.info("Size of validation is {} batches".format(len(svalidloader)))
    return trainloaders, testloaders, validloaders


def executeTraining(trainloaders, testloaders, tensor_weights, configuration):
    BATCHSIZE = configuration['batch']
    EPOCHS = configuration['epochs']
    assert(EPOCHS > 1)
    net, optimizer, criterion = makeNet(configuration)
    if configuration['mode']=='evaluate':
        return None, None, None, net
    BATCHRESET = 15
    PIX = configuration['pix']
    trainlosses, testlosses, batchlosses = [], [], []
    testbatchlosses = []
    mintestloss = float('inf')
    assert('fcn' in configuration['model'] or 'lstm' in configuration['model'])

    for epoch in tqdm(range(EPOCHS), desc='Epoch'):
        traindata = trainloaders
        tloss, iterations = 0, 0
        ibatchlosses = []
        for tdataset in tqdm(traindata, desc='Train Set'):
            tdloss, tdbatchlosses, iters = evaluateModel(tdataset, net, criterion, optimizer, tensor_weights, BATCHSIZE,
                                                         BATCHRESET, PIX, mode='train', strictbatches=configuration['model']=='lstm')
            tloss += tdloss
            iterations += iters
            batchlosses += tdbatchlosses
        tloss /= iterations
        lgr.debug("{} / {} : Average training loss over {} batches = {:10.4f}".format(epoch, EPOCHS, iterations, tloss))
        trainlosses.append(tloss)
        batchlosses += ibatchlosses

        testdata = testloaders
        tloss, iterations = 0, 0
        itestbatchlosses = []
        for tdataset in tqdm(testdata, desc='Test Set'):
            tdloss, tdbatchlosses, iters = evaluateModel(tdataset, net, criterion, optimizer, tensor_weights, BATCHSIZE,
                                                         BATCHRESET, PIX, mode='test', strictbatches=configuration['model']=='lstm')
            tloss += tdloss
            iterations += iters
            itestbatchlosses += tdbatchlosses
        tloss /= iterations
        if tloss < mintestloss:
            lgr.debug("New min test loss {} in epoch {} at loss {}, saving.".format(tloss, epoch, tloss))
            mintestloss = tloss
            torch.save({'state_dict': net.state_dict()}, os.path.join(configuration['outputdirectory'], 'model_{}.pth'.format(configuration['model'])))
        testlosses.append(tloss)
        testbatchlosses += itestbatchlosses

        if epoch > 1:
            last, now = trainlosses[-2], trainlosses[-1]
            diff = last - now
            if diff < 50:
                lgr.debug("Increasing loss, increasing rate")
                for param_group in optimizer.param_groups:
                    LR = param_group['lr']
                    param_group['lr'] = LR * 1.01

    return trainlosses, testlosses, batchlosses, net


if __name__ == "__main__":
    initialize()
    lgr.info("Loading data ...")
    svrg_data = LoadData(configuration['csv'], configuration['imageroot'], transform=ToTensor())
    evaldata(svrg_data, configuration)
    tensor_weights = computeWeights(svrg_data, configuration)
    listweights = configuration['weights']
    configuration['weights']=tensor_weights
    trainloaders, testloaders, validloaders = prepareData(svrg_data, configuration)
    # Train
    testlosses, trainlosses, batchlosses, net = executeTraining(trainloaders, testloaders, tensor_weights, configuration)
    # Eval
    ## Loading
    if configuration['mode'] == 'evaluate':
        checkpoint = torch.load(
        os.path.join(configuration['pretraineddir'], 'model.pth'.format(configuration['model'])))
        net.load_state_dict(checkpoint['state_dict'])
    else:
        lgr.info("Loading best network at epoch {}".format(np.argmin(testlosses)))

        checkpoint = torch.load(os.path.join(configuration['outputdirectory'], 'model_{}.pth'.format(configuration['model'])))
        net.load_state_dict(checkpoint['state_dict'])
    # Plot results
    lgr.info("Plotting results ...")
    evaldata = None
    if configuration['mode'] == 'validate':
        lgr.info("Validation mode selected, no changes can be made to arch.")
        evaldata = validloaders[0]
    elif configuration['mode'] == 'train' or configuration['mode'] == 'test':
        lgr.info("Train/Test mode selected")
        evaldata = testloaders[0]
    else:
        lgr.error("Invalid configuration mode !! :: {}".format(configuration['mode']))
        raise(AssertionError('Invalid configuration mode {}'.format(configuration['mode'])))
    quantifyLoss(net, trainlosses, testlosses, batchlosses, configuration)
    quantifyDistribution(net, trainloaders[0], configuration, 'Training_Distribution')
    gt, binned = quantifyDistribution(net, evaldata, configuration, 'Test_Distribution')
    np.save(os.path.join(configuration['outputdirectory'], 'gt.npy'), gt)
    np.save(os.path.join(configuration['outputdirectory'], 'binned.npy'), binned)

    C = configuration['trimclasses']
    gttr = gt.copy()
    binnedtr = binned.copy()
    gttr[gttr > C] = C
    binnedtr[binnedtr > C] = C

    np.save(os.path.join(configuration['outputdirectory'], 'gttr.npy'), gttr)
    np.save(os.path.join(configuration['outputdirectory'], 'binnedtr.npy'), binnedtr)

    quantifyPrediction(gttr, binnedtr, configuration)
    quantifyConfusion(gttr, binnedtr, configuration)
    configuration['weights']=listweights
    storeconfig(configuration)
    lgr.info("Done")
