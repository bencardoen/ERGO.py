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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
from scipy.stats import describe
from gconf import getlogger

lgr = getlogger()


def evaluate(gt, predicted):
    P, R, F, S = precision_recall_fscore_support(gt, predicted, average=None)
    res = []
    for (classid, (p, r, f, s)) in enumerate(zip(P, R, F, S)):
        res.append([classid, p, r, f, s])

    return pd.DataFrame(res, columns=['Class', 'Precision', 'Recall', 'F1', 'Support'])


def quantifyLoss(net, trainlosses, testlosses, batchlosses, configuration):
    Start = 4 if len(trainlosses) > 5 else 0
    width_fig, width_ax = plt.subplots() # 4-3
    d = {'trainloss': trainlosses[Start:], 'timetrain': [i for i, _ in enumerate(trainlosses[Start:])],
         'testloss': testlosses[Start:], 'timetest': [i for i, _ in enumerate(testlosses[Start:])],
         'batchlosses': batchlosses, 'timebatch': [i for i, _ in enumerate(batchlosses)]}
    sns.lineplot(x="timetrain", y="trainloss", data=d, label='Training', ax=width_ax)
    sns.lineplot(x="timetest", y="testloss", data=d, label='Testing', ax=width_ax)
    width_ax.set(xlabel='Epoch', ylabel='Loss', title='Loss function over Epochs.')
    width_fig.savefig(os.path.join(configuration['outputdirectory'], 'loss.png'))


def quantifyDistribution(net, loader, configuration, title):
    BATCHSIZE= configuration['batch']
    PIX = configuration['pix']
    outs = []
    lbs = []
    net.eval()
    lgr.info("Quantifying on {} batches".format(len(loader)))
    for j, tsample in enumerate(loader, 0):
        if len(tsample['image']) != BATCHSIZE:
            continue
        im = tsample['image']
        lb = tsample['label']
        labels = lb.cuda()
        images = im.cuda().view(-1, PIX ** 2)
        seqimages = images.view(1, -1, PIX ** 2)  # go from batch, PIX, PIX to batch, PIX*2
        net.init_hidden()
        outputs = net(seqimages)
        noutputs = outputs.view(-1, BATCHSIZE).data.cpu().numpy()
        outs.append(noutputs)
        lbs.append(labels.data.cpu().numpy())
    predicted = np.concatenate(outs, axis=None)
    gt = np.concatenate(lbs)
    binned = np.round(predicted)
    width_fig, width_ax = plt.subplots() # 4-3
    sns.distplot(binned, label='Predicted Class', kde=False, norm_hist=True, ax=width_ax)
    sns.distplot(predicted, label='Predicted Continuous Count', kde=True, norm_hist=True, ax=width_ax)
    sns.distplot(gt, label='Ground truth', kde=False, norm_hist=True, ax=width_ax)
    width_ax.legend()
    width_ax.set(xlabel='Class', ylabel='Frequency')
    width_ax.set_title(title)
    width_fig.savefig(os.path.join(configuration['outputdirectory'], '{}.png'.format(title)))
    lgr.info("Mean of prediction = {} vs mean of labels = {}".format(np.mean(predicted), np.mean(gt)))
    return gt, binned


def accMat(mat):
    return sum(np.diag(mat)) / mat.sum()


def classmse(gt, binned):
    classes = int(max(gt))
    predclasses = int(max(binned))
    lgr.info("Have {} GT classes and {} predicted classes".format(classes, predclasses))
    lgr.info("Have {.shape} GT and {.shape} predicted".format(gt, binned))
    assert(classes > 0)
    res = [[] for _ in range(classes+1)]
    assert(gt.shape[0] == binned.shape[0])
    for (g,b) in zip(gt, binned):
        res[int(g)].append((g-b)**2)
    ns, means, stds, skews, curtosiss = [], [], [], [], []
    for i,r in enumerate(res):
        if r == []:
            lgr.info("Empty class MSE for class {}".format(i))
            n, mM, mean, std, skew, curtosis = [float('NaN')], [float('NaN')], [float('NaN')], [float('NaN')], [float('NaN')], [float('NaN')]
            ns.append(n)
            means.append(mean)
            stds.append(std)
            skews.append(skew)
            curtosiss.append(curtosis)
        else:
            n, mM, mean, std, skew, curtosis = describe(r)
            ns.append(n)
            means.append(mean)
            stds.append(std)
            skews.append(skew)
            curtosiss.append(curtosis)
    return ns, means, stds, skews, curtosiss



def quantifyPrediction(gt, binned, configuration):
    ns, means, stds, skews, curtosiss = classmse(gt, binned)
    res = []
    for (classid, (nm, mu, st, sk, cu)) in enumerate(zip(ns, means, stds, skews, curtosiss)):
        res.append([classid, nm, mu[0], st[0], sk[0], cu[0]])
    pd.DataFrame(res, columns=['Class', 'N','Mean MSE', 'VAR MSE', 'SKEW', 'CURT']).to_csv(os.path.join(configuration['outputdirectory'], 'mse.csv'))
    # lgr.info("Means {}".format(means))
    kappa = cohen_kappa_score(gt, binned)
    mcc = matthews_corrcoef(gt, binned)
    bas = balanced_accuracy_score(gt, binned, sample_weight=None, adjusted=False)
    basadj = balanced_accuracy_score(gt, binned, sample_weight=None, adjusted=True)
    lgr.info("Kappa {0:.2f} MCC {1:.2f} BASA {2:.2f} BAS {3:.2f}".format(kappa, mcc, basadj, bas))
    pd.DataFrame([[kappa, mcc, basadj, bas]], columns=['kappa', 'mcc', 'balanced_accuracy_adj', 'balanced_accuracy']).to_csv(os.path.join(configuration['outputdirectory'], 'kappa.csv'))
    P, R, F, S = precision_recall_fscore_support(gt, binned, average='weighted')
    weighted = pd.DataFrame([[P, R, F]], columns=['Precision', 'Recall', 'F1'])
    weighted.to_csv(os.path.join(configuration['outputdirectory'], 'prec_rec_f1_weighted.csv'))
    pdf = evaluate(gt, binned)
    pdf.to_csv(os.path.join(configuration['outputdirectory'], 'prec_rec_f1_unweighted.csv'))


def quantifyConfusion(gt, binned, configuration):
    for c in [1, 2, float('inf')]:
        for normalize in [True, False]:
            width_fig, width_ax = plt.subplots()  # 4-3
            gtsp = gt.copy()
            binnedsp = binned.copy()
            if c != float('inf'):
                gtsp[gt >= c] = c
                binnedsp[binnedsp >= c] = c
            labels = None
            if c != float('inf'):
                labels = ['Noise', 'Signal'] if c == 1 else ['0', '1', '>1']
            else:
                labels = [str(i) for i in range(configuration['trimclasses'])]
                labels += [">= {}".format(configuration['trimclasses'])]
            cmn = confusion_matrix(gtsp, binnedsp)
            acc = None
            lgr.debug("Accuracy = {}".format(acc))
            if normalize:
                cmn = cmn.astype('float') / cmn.sum(axis=1)[:, np.newaxis]
                acc = accMat(cmn)
                sns.heatmap(cmn, annot=True if c <= 8 else False, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=labels,
                             yticklabels=labels, cbar=True, cbar_kws={'label': 'Frequency'}, ax=width_ax)
            else:
                sns.heatmap(cmn, annot=True if c <= 8 else False, fmt=".2f", cmap="Blues",xticklabels=labels,
                            yticklabels=labels, cbar=True, cbar_kws={'label': 'Frequency'}, ax=width_ax)
                acc = accMat(cmn)
            #             ax.invert_yaxis()
            width_ax.tick_params(axis='x', rotation=45)
            width_ax.tick_params(axis='y', rotation=45)
            title = "Confusion Matrix For Classes {0} Accuracy {1:.2f} {2}".format(c, acc, 'Normalized' if normalize else '')
            plt.suptitle(title)
            width_fig.savefig(os.path.join(configuration['outputdirectory'], '{}.png'.format(title)))



def plotfigures(data, methods, gt, outdir):
    classes = [2, 3, float('inf')]
    width_fig, width_ax = plt.subplots(len(classes),len(methods),figsize=(20,14)) # 4-3
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.6)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    NM = len(methods)
    fname = None
    for j, (row, c) in enumerate(zip(width_ax, classes)):
        lgr.info("Dealing with class {}".format(c))
        for i, (method, ax) in enumerate(zip(methods,row)):
            binnedsp = data[method].copy()
            gtsp = gt.copy()
            labels = None
            if c == 2:
                gtsp = gt.copy()
                gtsp[gtsp > c-1] = c-1
                binnedsp[binnedsp > c-1] = c-1
                labels = ['Noise', 'Signal']
            elif c == 3:
                gtsp = gt.copy()
                gtsp[gtsp > c-1] = c-1
                binnedsp[binnedsp > c-1] = c-1
                labels = ['Noise', 'Sparse Signal', 'Dense Signal']
            else:
                mc = max(gt)
                binnedsp[binnedsp > mc[0]] = mc[0] # Meaningless to predict things that do not exist
                labels = [str(i) for i in range(int(mc[0])+1)]
            cmsp = confusion_matrix(gtsp, binnedsp)
            cmn = cmsp.astype('float') / cmsp.sum(axis=1)[:, np.newaxis]
            ax = sns.heatmap(cmn, annot=True if c <= 3 else False, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=labels,
                        yticklabels=labels, ax = ax, cbar= (i==NM-1), cbar_kws={'label': 'Frequency'})
#             ax.invert_yaxis()
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=45)
            if j == 0:
                if i != NM-1:
                    ax.set_title("{}".format(method))
                else:
                    ax.set_title("{} (ours)".format(method))
            ax.set(xlabel='Predicted Count')
            if i == 0:
                ax.set(ylabel='True Count')
    fname = os.path.join(outdir,'conf_mat.png')
    lgr.info("Figure will be saved to {}".format(fname))
    plt.suptitle("Confusion Matrix")
    width_fig.savefig(fname, dpi=300)
