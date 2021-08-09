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
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from skimage import io
from torch.utils.data import Dataset
from skimage import img_as_float
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from gconf import getlogger
from copy import deepcopy

lgr = getlogger()


class LoadData(object):
    def __init__(self, csv_file, root_dir, transform=None, data=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        frame = pd.read_csv(csv_file)
        lgr.info("Reading from dir {} with annotation file {}".format(root_dir, csv_file))
        self.gt_frame = frame.sort_values(['framenr', 'roinr'], ascending=[True, True])
        self.root_dir = root_dir
        assert(os.path.exists(root_dir))
        self.mapping = []
        self._loadimgs()

    def _loadimgs(self):
        self.ftable = [0 for _ in range(max(self.gt_frame['count']+1))]
        N = len(self.gt_frame.index)
        for (j, i) in enumerate(tqdm(self.gt_frame.itertuples(), total=N, desc='File')):
            framenr, roinr = i.framenr, i.roinr
            roil = roinr

            lgr.debug("FNR {} RNR {}".format(framenr, roinr))
            if roinr == 0:
                roil = 'negative'
            img_name = 'frame_{}_{}.tiff'.format(framenr, roil)
            fp = os.path.join(self.root_dir, img_name)
            if os.path.exists(fp):
                image = io.imread(os.path.join(self.root_dir, img_name))
                self.mapping.append([img_as_float(image), framenr, roinr, i.count])
                self.ftable[i.count] += 1
            else:
                lgr.error("No such file {}".format(img_name))
                continue
        lgr.info("Completed a total of {} images".format(j+1))

    @property
    def data(self):
        return self.mapping

    @property
    def classes(self):
        return len(self.frequency)

    @property
    def frequency(self):
        """
        :return: Normalized frequency table.
        """
        return [l / sum(self.ftable) for l in self.ftable]

    def __len__(self):
        return len(self.mapping)

class LoadData(object):
    def __init__(self, csv_file, root_dir, transform=None, data=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        frame = pd.read_csv(csv_file)
        lgr.info("Reading from dir {} with annotation file {}".format(root_dir, csv_file))
        self.gt_frame = frame.sort_values(['framenr', 'roinr'], ascending=[True, True])
        self.root_dir = root_dir
        assert(os.path.exists(root_dir))
        self.mapping = []
        self._loadimgs()
        # Transform

    def _loadimgs(self):
        self.ftable = [0 for _ in range(max(self.gt_frame['count']+1))]
        N = len(self.gt_frame.index)
        for (j, i) in enumerate(tqdm(self.gt_frame.itertuples(), total=N, desc='File')):
            framenr, roinr = i.framenr, i.roinr
            roil = roinr

            lgr.debug("FNR {} RNR {}".format(framenr, roinr))
            if roinr == 0:
                roil = 'negative'
            img_name = 'frame_{}_{}.tiff'.format(framenr, roil)
            fp = os.path.join(self.root_dir, img_name)
            if os.path.exists(fp):
                image = io.imread(os.path.join(self.root_dir, img_name))
                self.mapping.append([img_as_float(image), framenr, roinr, i.count])
                self.ftable[i.count] += 1
            else:
                lgr.error("No such file {}".format(img_name))
                continue
        lgr.info("Completed a total of {} images".format(j+1))

    @property
    def data(self):
        return self.mapping

    @property
    def classes(self):
        return len(self.frequency)

    @property
    def frequency(self):
        """
        :return: Normalized frequency table.
        """
        return [l / sum(self.ftable) for l in self.ftable]

    def __len__(self):
        return len(self.mapping)


    def duplicate(self):
        lgr.info("Duplicating data")
        N = len(self.mapping)
        cm = deepcopy(self.mapping) + deepcopy(self.mapping)
        self.mapping = cm
        lgr.info("Old mapping {} , New mapping = {}".format(N, len(self)))


    def adjust(self, subtract):
        lgr.info("Subtracting {}".format(subtract))
        for i,_ in enumerate(self.mapping):
            m = np.mean(self.mapping[i][0])
            self.mapping[i][0] -= subtract
            m2 = np.mean(self.mapping[i][0])
            lgr.info("Mean {} -> {}".format(m, m2))



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        final = {}

        final['image'] = torch.from_numpy(image).type(torch.FloatTensor)
        final['label'] = torch.from_numpy(np.array([label])).type(torch.FloatTensor)
        return final


class SVRGDataset(Dataset):

    def __init__(self, data, indices=None, transform=None, seed=None, augment=None, augmentlive=None):
        """
        :augment str: Augment data: If 'balanced', augment all classes except the most frequent one, if 'sequential', augment but keep seqeunce intact
        """
        self.mapping = data
        self.old = None
        self.indices = indices
        self.seed = seed
        if not transform:
            raise 42
        self.transform = transform
        if indices:
            self.apply_indices()
        if augment is not None:
            lgr.info("Using augmentation method {}".format(augment))
            self.augmentmethod = augment
            self.augment()

    @property
    def frequency(self):
        f = {}
        for m in self.mapping:
            l = m[-1]
            if l in f:
                f[l] += 1
            else:
                f[l] = 1
        fl = [f[k] for k in sorted(f.keys())]
        fs = sum(fl)
        fn = [fle/fs for fle in fl]
        return fn

    def augment(self):
        ftable = self.frequency
        probability = [1-f for f in ftable]
        lgr.debug("Probability table {}".format(probability))
        nmap = []
        cnt = 0
        lgr.debug("Augmenting data ...")
        for m in tqdm(self.old, desc='Image'):
            label = m[-1]
            if self.augmentmethod == 'balanced':
                nmap.append(m)
                if label != 2:
                    cnt += 3
                    nmap.append([np.rot90(m[0]).copy(),m[1],m[2],label])
                    nmap.append([np.rot90(np.rot90(m[0])).copy(),m[1],m[2],label])
                    nmap.append([np.rot90(np.rot90(np.rot90(m[0])).copy()).copy(),m[1],m[2],label])
            elif self.augmentmethod == '90':
                if label != 2:
                    nmap.append([np.rot90(m[0]).copy(),m[1],m[2],label])
                    cnt += 1
            elif self.augmentmethod == 'noise':
                nmap.append(m)
                if label == 0:
                    nmap.append([np.rot90(m[0]).copy(),m[1],m[2],label])
                    cnt += 1
            elif self.augmentmethod == '180':
                if label != 2:
                    nmap.append([np.rot90(np.rot90(m[0])).copy(),m[1],m[2],label])
                    cnt += 1
            elif self.augmentmethod == '270':
                if label != 2:
                    nmap.append([np.rot90(np.rot90(np.rot90(m[0])).copy()).copy(),m[1],m[2],label])
                    cnt += 1
        self.mapping = nmap
        lgr.debug("Added {} images total of {}".format(cnt, len(nmap)))

    def apply_indices(self):
        lgr.debug("Validating new indices {}".format(len(self.indices)))
        self.old = deepcopy(self.mapping)
        mp = [self.mapping[i] for i in self.indices]
        lgr.debug("New data is {}".format(len(mp)))
        self.mapping = mp

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, jdx):
        sample = {}
        sample['image'] = self.mapping[jdx][0]
        sample['label'] = self.mapping[jdx][-1]

        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample


def findframe(roiindex, mapping):
    oldframe = mapping[roiindex][1]
    lgr.debug("Starting at index {} to find new frames, current is {}".format(roiindex, oldframe))
    N = len(mapping)
    while roiindex < N:
        m = mapping[roiindex]
        frame = m[1]
        if oldframe != frame:
            lgr.debug("Index {} first to change frames from {} to {}".format(roiindex, oldframe, frame))
            return roiindex
            break
        roiindex += 1
    return None


def findframerange(begin, end, mapping):
    return mapping[begin][1], mapping[end][1]


def evaluate(gt, predicted):
    P, R, F, S = precision_recall_fscore_support(gt, predicted, average=None)
    res = []
    for (classid, (p, r, f, s)) in enumerate(zip(P, R, F, S)):
        res.append([classid, p, r, f, s])

    return pd.DataFrame(res, columns=['Class','Precision', 'Recall', 'F1', 'Support'])


def findpredicted(methodname, framenr, soadata):
    _df = soadata[methodname]
    mask = _df['frame_id'] == framenr
    values = _df[mask]
    result = []
    for _,value in values.iterrows():
        result.append([int(np.round(value['x_nm']/100)+1), int(np.round(value['y_nm']/100)+1)])
    return result
