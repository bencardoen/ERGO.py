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
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from gconf import getlogger

lgr = getlogger()


class SVGLoss(nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction='mean', weights=None, toint=False, batchweights=False, losstype='mse'):
        super(SVGLoss, self).__init__(size_average, reduce, reduction)
        if weights is not None:
            wstr = ' '.join('{0:.2f}'.format(el) for _, el in enumerate(weights))
            lgr.info("Weights are {}".format(wstr))
        self.weights = weights
        self.batchweights = batchweights
        self.toint = toint
        self.loss = self._mseloss if 'mse' in losstype else self._exploss
        lgr.info("Loss : \n\tLoss function = {} \n\tInt Mode = {} \n\tBatch Weights = {}".format(self.loss, self.toint, self.batchweights))

    def weightedloss(self, predicted, target, weights=None, toint=False):
        assert (predicted.size() == target.size())
        if weights is not None:
            assert (len(target.size()) == len(weights.size()))

        if toint:
            predicted = predicted.clone()
            predicted[torch.abs(predicted - target) < 0.5] = target[torch.abs(predicted - target) < 0.5]
        _ms = self.loss(predicted, target)
        if weights is not None:
            return _ms * weights[target.long()]
        else:
            return _ms

    @staticmethod
    def _mseloss(predicted, target):
        return (predicted - target) ** 2

    @staticmethod
    def _exploss(predicted, target, sigma=0.5):
        return 1 - torch.exp(- torch.abs(predicted - target) / (2 * sigma ** 2))

    def forward(self, pred, trg):
        target = trg.squeeze(1)
        target = target.clone()
        predicted = pred.squeeze(0).squeeze(1).clone()
        lgr.debug("Y size {} -> {}".format(trg.size(), target.size()))
        lgr.debug("Y' size {} -> {}".format(pred.size(), predicted.size()))
        fwd = None
        if self.batchweights:
            freq = torch.zeros(20).cuda()
            for yp in target:
                freq[yp.long()] += 1
            freq = freq / sum(freq)
            fweights = freq.clone()
            for i, f in enumerate(freq):
                if f != 0:
                    fweights[i] = 2 - f
            lgr.debug("Batch weights {}".format(fweights))
            fwd = self.weightedloss(predicted, target, fweights, self.toint)
        else:
            fwd = self.weightedloss(predicted, target, self.weights, self.toint)
        return fwd


class FCNSVRG(nn.Module):
    def __init__(self, PIX):
        super(FCNSVRG, self).__init__()
        self.fc1 = nn.Linear(PIX**2, (PIX-1)**2)
        self.fc2 = nn.Linear((PIX-1)**2, (PIX-2)**2)
        self.fc3 = nn.Linear((PIX-2)**2, 1)

    def init_hidden(self):
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LSTMSVRG(nn.Module):

    def __init__(self, inputdim, hiddendim, batchsize, seqsize, pix, initstrategy='zeros'):
        super(LSTMSVRG, self).__init__()
        self.hiddendim = hiddendim
        self.lstm = nn.LSTM(inputdim, self.hiddendim)
        self.batchsize = batchsize
        self.seqsize = seqsize
        self.pix = pix
        self.initstrategy=initstrategy

        self.fc1 = nn.Linear(self.hiddendim, (self.pix-2)**2)
        self.fc2 = nn.Linear((self.pix-2)**2, 1)
        lgr.info("LSTM with input dim {} hidden dim {} output {}".format(inputdim, hiddendim, 1))
        (h0, h1) = (torch.zeros(self.seqsize, self.batchsize, self.hiddendim),
                torch.zeros(self.seqsize, self.batchsize, self.hiddendim))
        self.hidden = (h0.cuda(), h1.cuda())

    def init_hidden(self):
        (h0, h1) = (None, None)
        if self.initstrategy == 'zero':
            (h0, h1) = (torch.zeros(self.seqsize, self.batchsize, self.hiddendim),torch.zeros(self.seqsize, self.batchsize, self.hiddendim))
        elif self.initstrategy == 'random':
            (h0, h1) = (torch.rand(self.seqsize, self.batchsize, self.hiddendim),torch.rand(self.seqsize, self.batchsize, self.hiddendim))
        self.hidden = (h0.cuda(), h1.cuda())

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        x = torch.nn.functional.relu(self.fc1(lstm_out))
        x = torch.nn.functional.relu(self.fc2(x))
        return x


def makeNet(configuration):
    BATCHSIZE=configuration['batch']
    LR = configuration['lr']
    PIX = configuration['pix']
    assert('model' in configuration)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = None
    if configuration['model'] == 'lstm':
        net = LSTMSVRG(PIX**2, (PIX-1)**2, BATCHSIZE, 1, PIX, configuration['initstrategy'])
    elif configuration['model'] == 'fcn':
        net = FCNSVRG(PIX=PIX)
    else:
        lgr.error("No such model {}".format(configuration['model']))
        raise(AssertionError('No such model {}'.format(configuration['model'])))
    net.to(device)
    tensor_weights = configuration['weights']
    criterion = SVGLoss(reduce=False, weights=tensor_weights, toint='int' in configuration['loss'], batchweights=configuration['batchweights'], losstype=configuration['loss'])
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    return net, optimizer, criterion


def evaluateModel(data, net, criterion, optimizer, weights, batchsize, batchreset, PIX, mode='train', strictbatches=True):
    tloss = 0.0
    running_loss = 0.0
    batchlosses = []
    if mode == 'train':
        net = net.train()
    else:
        net = net.eval()
    for i, sample in enumerate(tqdm(data, desc='Batch')):
        if strictbatches and len(sample['image']) != batchsize:
            continue
        if mode == 'train':
            net.zero_grad()
        im = sample['image']
        lb = sample['label']

        images, labels = im.cuda(), lb.cuda()
        seqimages = images.view(1, -1, PIX**2)
        net.init_hidden()
        outputs = net(seqimages)
        tensorloss = criterion(outputs, labels)
        loss = tensorloss.sum()
        if mode == 'train':
            loss.backward()
            optimizer.step()
        ls = loss.item()
        tloss += ls
        running_loss += ls
        if i % batchreset == batchreset-1:
            batchlosses.append(running_loss/batchreset)
            running_loss = 0.0
            if mode == 'train':
                for param_group in optimizer.param_groups:
                    LR = param_group['lr']
                    param_group['lr'] = LR*0.99995

    return tloss, batchlosses, i
