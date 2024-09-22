# define core utility functions for loading point cloud datasets

import os
import sys
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pythae.data.datasets import DatasetOutput


class PointDataReg(Dataset):
    def __init__(self, root, fluo_channel, training_dates, split='train', point_features=None, npoints=4096, r_prob=0.75,
                 shear_prob=1.0, cut_prob=0.5):
        self.root = root
        self.point_features = point_features
        if self.point_features is None:
            self.point_features = []
        self.fluo_channel = fluo_channel
        # self.area_nums = area_nums  # i.e. '1-4' # areas 1-4
        self.split = split.lower()  # use 'test' in order to bypass augmentations
        self.npoints = npoints  # use  None to sample all the points
        self.r_prob = r_prob  # probability of rotation
        self.shear_prob = shear_prob
        self.cut_prob = cut_prob
        all_paths = sorted(glob(os.path.join(root, '**/*.csv'), recursive=True))

        self.data_paths = [p for p in all_paths if any(d in p for d in training_dates)]


    def __getitem__(self, idx):


        feature_cols = self.point_features

        # read data from hdf5
        space_data = pd.read_csv(self.data_paths[idx])
        points_raw = space_data.loc[:, ["Z", "Y", "X"] + feature_cols].to_numpy().astype(np.float64)

        # points_raw = space_data.loc[:, ["X", "Y", "Z", self.fluo_channel]].to_numpy()
        if self.fluo_channel is not None:# xyz points
            targets = np.reshape(space_data.loc[:, self.fluo_channel].to_numpy(), (points_raw.shape[0], 1))
        else:
            targets = -1*np.ones((points_raw.shape[0], 1))

        # take subsection of full point cloud
        # if self.split != "test":
        #     points, targets, indices = self.random_cuts(points_raw, targets)
        # down sample point cloud
        # if self.npoints:
        points, targets, indices = self.resample_with_cuts(points_raw, targets)
        points_raw = points_raw[indices]



        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add random rotation to the point cloud with probability
            points = self.random_rotate(points)

            # add random shear
            points = self.random_shear(points)

            # Normalize Point Cloud to (0, 1)
            points = self.normalize_points(points)

            # add N(0, 1/100) noise
            points[:, 0:3] += np.random.normal(0., 0.01, points[:, 0:3].shape)

        else:
            # Normalize Point Cloud to (0, 1)
            points = self.normalize_points(points)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.float32)

        out = DatasetOutput(data=points, label=targets, path=self.data_paths[idx], point_indices=indices, raw_data=points_raw)

        return out

    def resample(self, points, targets):
        if len(points) >= self.npoints:
            # raise Exception("This should not happen")
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are fewer points than the desired number
            choice = np.random.choice(len(points), len(points), replace=False)

        points = points[choice, :]
        targets = targets[choice]

        return points, targets, choice

    def resample_with_cuts(self, points, targets):

        if (self.split != 'test') & (np.random.rand() <= self.cut_prob):

            dim = np.random.randint(0, 3, 1)[0]

            # randomly sample start
            d_rank = np.argsort(points[:, dim])
            start_max = np.ceil(points.shape[0] / 2).astype(int)
            start_i = np.random.choice(range(start_max), 1, replace=False)[0]
            if np.random.rand() < 0.5:
                keep_indices = d_rank[start_i:]
            else:
                if start_i > 0:
                    keep_indices = d_rank[:-start_i]
                else:
                    keep_indices = d_rank

            points = points[keep_indices, :]
            targets = targets[keep_indices]

        else:
            keep_indices = np.arange(points.shape[0])

        # standardize cloud size
        if len(points) >= self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are fewer points than the desired number
            choice1 = np.random.choice(len(points), len(points), replace=False)
            choice2 = np.random.choice(len(points), self.npoints-len(points), replace=True)
            choice = np.concatenate((choice1, choice2))

            # n_points = len(points)
            # choice = np.tile(range(n_points), int(np.ceil(self.npoints/n_points)))
            # choice = choice[:self.npoints]
        points = points[choice, :]
        targets = targets[choice]

        return points, targets, keep_indices[choice]

    def random_rotate(self, points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix

        if np.random.rand() <= self.r_prob:
            phi = np.random.uniform(-np.pi, np.pi)
            theta = np.random.uniform(-np.pi, np.pi)
            psi = np.random.uniform(-np.pi, np.pi)

            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)]])

            rot_y = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])

            rot_z = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]])

            rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))

            points_temp = points[:, 0:3][:, ::-1]
            points[:, 0:3] = np.matmul(points_temp, rot)[:, ::-1]
            # points[:, 0:3] = points[:, 0:3] - np.min(points[:, 0:3], axis=0)
            # check for nucleus shape info
            if "pZ_nn" in self.point_features:
                ind = self.point_features.index("pZ_nn")
                feat_temp = points[:, ind+3:ind+6][:, ::-1]
                points[:, ind+3:ind+6] = np.abs(np.matmul(feat_temp, rot)[:, ::-1])
                # points[:, ind + 3:ind + 6] = points[:, ind+3:ind+6] - np.min(points[:, ind+3:ind+6], axis=0)

        return points

    def random_shear(self, points):

        ub = 1.5
        lb = 0.5
        if np.random.rand() <= self.shear_prob:
            shear_vec = np.random.uniform(low=lb, high=ub, size=(1, 3))
            points[:, 0:3] = np.multiply(points[:, 0:3], shear_vec)

        return points

    @staticmethod
    def random_cuts(points):
        dim = np.random.randint(0, 3, 1)[0]

        # randomly sample start
        d_rank = np.argsort(points[:, dim])
        start_max = np.ceil(points.shape[0]/2).astype(int)
        start_i = np.random.choice(range(start_max), 1, replace=False)
        if np.random.rand() < 0.5:
            keep_points = d_rank[start_i:]
        else:
            keep_points = d_rank[:-start_i]

        points = points[keep_points, :]
        return points


    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points[:, 0:3] = points[:, 0:3] - points[:, 0:3].min(axis=0)
        points[:, 0:3] /= points[:, 0:3].max(axis=0)

        return points

    def __len__(self):
        return len(self.data_paths)


class PointData(Dataset):
    def __init__(self, root, split='train', npoints=4096, r_prob=0.25, fluo_channel=None):
        self.root = root
        self.fluo_channel = fluo_channel
        # self.area_nums = area_nums  # i.e. '1-4' # areas 1-4
        self.split = split.lower()  # use 'test' in order to bypass augmentations
        self.npoints = npoints  # use  None to sample all the points
        self.r_prob = r_prob  # probability of rotation

        # # glob all hdf paths
        # areas = glob(os.path.join(root, f'Area_[{area_nums}]*'))
        #
        # # check that datapaths are valid, if not raise error
        # if len(areas) == 0:
        #     raise FileNotFoundError("NO VALID FILEPATHS FOUND!")
        #
        # for p in areas:
        #     if not os.path.exists(p):
        #         raise FileNotFoundError(f"PATH NOT VALID: {p} \n")

        # get all datapaths
        # self.data_paths = []
        # for area in areas:
        self.data_paths = sorted(glob(os.path.join(root, '**/*.csv'), recursive=True))

        # get unique space identifiers (area_##\\spacename_##_)
        # self.space_ids = []
        # for fp in self.data_paths:
        #     area, space = fp.split('\\')[-2:]
        #     space_id = '\\'.join([area, '_'.join(space.split('_')[:2])]) + '_'
        #     self.space_ids.append(space_id)
        #
        # self.space_ids = list(set(self.space_ids))

    def __getitem__(self, idx):
        # read data from hdf5
        space_data = pd.read_csv(self.data_paths[idx])
        if self.fluo_channel is None:
            points_raw = space_data.loc[:, ["X", "Y", "Z"]].to_numpy()
        else:
            points_raw = space_data.loc[:, ["X", "Y", "Z", self.fluo_channel]].to_numpy()
        if "fin_label_curr" in space_data.columns:# xyz points
            targets = np.reshape(space_data.loc[:, "fin_label_curr"].to_numpy(), (points_raw.shape[0], 1))  # integer categories
        else:
            targets = -1*np.ones((points_raw.shape[0], 1))
        # down sample point cloud
        # if self.npoints:
        points, targets, indices = self.downsample(points_raw, targets)
        points_raw = points_raw[indices]
        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            # points += np.random.normal(0., 0.01, points.shape)

            # add random rotation to the point cloud with probability
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                points = self.random_rotate(points)

        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        out = DatasetOutput(data=points, label=targets, path=self.data_paths[idx], point_indices=indices, raw_data=points_raw)

        return out


    def get_random_partitioned_space(self):
        ''' Obtains a Random space. In this case the batchsize would be
            the number of partitons that the space was separated into.
            This is a special function for testing.
            '''

        # get random space id
        idx = random.randint(0, len(self.space_ids) - 1)
        space_id = self.space_ids[idx]

        # get all filepaths for randomly selected space
        space_paths = []
        for fpath in self.data_paths:
            if space_id in fpath:
                space_paths.append(fpath)

        # assume npoints is very large if not passed
        if not self.npoints:
            self.npoints = 20000

        points = np.zeros((len(space_paths), self.npoints, 3))
        targets = np.zeros((len(space_paths), self.npoints))

        # obtain data
        for i, space_path in enumerate(space_paths):
            space_data = pd.read_hdf(space_path, key='space_slice').to_numpy()
            _points = space_data[:, :3]  # xyz points
            _targets = space_data[:, 3]  # integer categories

            # downsample point cloud
            _points, _targets = self.downsample(_points, _targets)

            # add points and targets to batch arrays
            points[i] = _points
            targets[i] = _targets

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets

    # def downsample(self, points, targets):
    #     if len(points) > self.npoints:
    #         raise Exception("This should not happen")
    #         choice = np.random.choice(len(points), self.npoints, replace=False)
    #     else:
    #         # case when there are fewer points than the desired number
    #         # choice = np.random.choice(len(points), self.npoints, replace=True)
    #         n_points = len(points)
    #         choice = np.tile(range(n_points), int(np.ceil(self.npoints/n_points)))
    #         choice = choice[:self.npoints]
    #     points = points[choice, :]
    #     targets = targets[choice]
    #
    #     return points, targets, choice

    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]])

        rot_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))

        return np.matmul(points, rot_z)

    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points

    def __len__(self):
        return len(self.data_paths)