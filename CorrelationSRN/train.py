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

import torch

from data.datasets import CorrelationDataSet
from network.correlation_srn import CorrelationSRN


if __name__ == '__main__':
    batch_size = 256
    num_iterations = 100
    eval_step = 1
    device = torch.device('cuda')

    print('Initializing model...')
    model = CorrelationSRN(
        '/media/christoph/Elements/Datasets/SRNs/2023-09-06/pearson_u_hash30_rand300_ch128_7_6_23.zip',
        #'/media/christoph/Elements/Datasets/SRNs/2023-09-06/model-std.zip',
        use_tcnn=False)
    print('Opening data set...')
    dataset = CorrelationDataSet(
        '/home/christoph/datasets/Necker/nc/necker_t5_tk_u.nc',
        'u', batch_size=batch_size)

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print('Starting optimization...')
    running_loss = 0.0
    for i in range(num_iterations):
        optimizer.zero_grad()

        Xc, Yc, gt = dataset.sample_batch(model, use_bos=True)
        outputs = model(Xc, Yc)

        loss = loss_function(outputs, gt)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if i % eval_step == eval_step - 1:
            train_loss_avg = running_loss / eval_step
            print(f'Train loss: {train_loss_avg}')
            running_loss = 0.0

    print('Saving model...')
    model.save_network('/media/christoph/Elements/Datasets/SRNs/2023-09-09/network-2023-09-09.zip')
    print('Quitting program...')
