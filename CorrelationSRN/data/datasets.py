#
# BSD 2-Clause License
#
# Copyright (c) 2023, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
from numba import jit
import netCDF4
import torch
import pycoriander
import pycorlimbo


#@jit(nopython=False)
def convert_point_to_coords(x, y, z, xs, ys, zs):
    coords = torch.empty(3, dtype=torch.float, device='cpu')
    coords[0] = 2.0 * float(x) / float(xs - 1) - 1.0
    coords[1] = 2.0 * float(y) / float(ys - 1) - 1.0
    coords[2] = 2.0 * float(z) / float(zs - 1) - 1.0
    #return [
    #    2.0 * float(x) / float(xs - 1) - 1.0,
    #    2.0 * float(y) / float(ys - 1) - 1.0,
    #    2.0 * float(z) / float(zs - 1) - 1.0
    #]


#@jit(nopython=True)
def convert_sample_locations(grid_locations: torch.Tensor, data_all, xs, ys, zs, es):
    num_samples = grid_locations.size()[0]
    X = torch.empty((num_samples, es), dtype=torch.float, device='cpu')
    Y = torch.empty((num_samples, es), dtype=torch.float, device='cpu')
    Xc = torch.empty((num_samples, 3), dtype=torch.float, device='cpu')
    Yc = torch.empty((num_samples, 3), dtype=torch.float, device='cpu')
    for sample_idx in range(num_samples):
        xi = grid_locations[sample_idx, 0].item()
        yi = grid_locations[sample_idx, 1].item()
        zi = grid_locations[sample_idx, 2].item()
        xj = grid_locations[sample_idx, 3].item()
        yj = grid_locations[sample_idx, 4].item()
        zj = grid_locations[sample_idx, 5].item()
        X[sample_idx, :] = torch.tensor(data_all[:, zi, yi, xi], dtype=torch.float, device='cpu')
        Y[sample_idx, :] = torch.tensor(data_all[:, zj, yj, xj], dtype=torch.float, device='cpu')
        #ci = convert_point_to_coords(xi, yi, zi, xs, ys, zs)
        #cj = convert_point_to_coords(xj, yj, zj, xs, ys, zs)
        #for i in range(3):
        #    Xc[sample_idx, i] = ci[i]
        #    Yc[sample_idx, i] = cj[i]
        Xc[sample_idx, 0] = 2.0 * float(xi) / float(xs - 1) - 1.0
        Xc[sample_idx, 1] = 2.0 * float(yi) / float(ys - 1) - 1.0
        Xc[sample_idx, 2] = 2.0 * float(zi) / float(zs - 1) - 1.0
        Yc[sample_idx, 0] = 2.0 * float(xj) / float(xs - 1) - 1.0
        Yc[sample_idx, 1] = 2.0 * float(yj) / float(ys - 1) - 1.0
        Yc[sample_idx, 2] = 2.0 * float(zj) / float(zs - 1) - 1.0
    return X, Y, Xc, Yc


class CorrelationDataSet:
    def __init__(self, data_path, variable_name, batch_size, num_init_points=10, num_bos_iterations=50):
        ncfile = netCDF4.Dataset(data_path, 'r')
        self.batch_size = batch_size
        self.data_all = ncfile[variable_name][:, :, :, :].filled(fill_value=np.nan)
        self.es = self.data_all.shape[0]
        self.zs = self.data_all.shape[1]
        self.ys = self.data_all.shape[2]
        self.xs = self.data_all.shape[3]

        self.settings = pycorlimbo.BayOptSettings()
        self.settings.xs = self.xs
        self.settings.ys = self.ys
        self.settings.zs = self.zs
        self.settings.num_init_samples = num_init_points
        self.settings.num_iterations = num_bos_iterations
        #settings.alpha = 0.5
        self.train_locations = torch.empty((batch_size, 6), dtype=torch.int32)
        valid_locations = np.argwhere(~np.isnan(self.data_all[:, :, :, :]).any(axis=0))
        valid_locations = np.flip(valid_locations, axis=1).copy()
        self.valid_locations = torch.tensor(valid_locations)

    def sample_batch(self, model, use_bos=True):
        if use_bos:
            xs = self.xs
            ys = self.ys
            zs = self.zs
            es = self.es
            data_all = self.data_all
            def sample_correlation_function(query, result):
                num_samples = query.size()[0]
                X, Y, Xc, Yc = convert_sample_locations(query, data_all, xs, ys, zs, es)
                Xc = Xc.to('cuda')
                Yc = Yc.to('cuda')
                values_gt = pycoriander.pearson_correlation(X, Y).cpu()
                with torch.no_grad():
                    values_srn = model(Xc, Yc).cpu()
                for sample_idx in range(num_samples):
                    if np.isnan(values_gt[sample_idx].item()):
                        result[sample_idx] = 0.0
                    else:
                        result[sample_idx] = abs(values_gt[sample_idx] - values_srn[sample_idx]).item()

            pycorlimbo.optimize_multi_threaded(
                self.settings, self.train_locations, sample_correlation_function)
        else:
            #self.train_locations[:, 0] = torch.randint(0, self.xs - 1, (self.batch_size,))
            #self.train_locations[:, 1] = torch.randint(0, self.ys - 1, (self.batch_size,))
            #self.train_locations[:, 2] = torch.randint(0, self.zs - 1, (self.batch_size,))
            #self.train_locations[:, 3] = torch.randint(0, self.xs - 1, (self.batch_size,))
            #self.train_locations[:, 4] = torch.randint(0, self.ys - 1, (self.batch_size,))
            #self.train_locations[:, 5] = torch.randint(0, self.zs - 1, (self.batch_size,))
            idx0 = torch.randint(0, self.valid_locations.shape[0], (self.batch_size,))
            idx1 = torch.randint(0, self.valid_locations.shape[0], (self.batch_size,))
            self.train_locations[:, 0:3] = self.valid_locations[idx0[:]]
            self.train_locations[:, 3:6] = self.valid_locations[idx1[:]]

        X, Y, Xc, Yc = convert_sample_locations(
            self.train_locations, self.data_all, self.xs, self.ys, self.zs, self.es)
        Xc = Xc.to('cuda')
        Yc = Yc.to('cuda')
        values_gt = pycoriander.pearson_correlation(X, Y)
        values_gt = values_gt.to('cuda').reshape((self.batch_size, 1))

        return Xc, Yc, values_gt
