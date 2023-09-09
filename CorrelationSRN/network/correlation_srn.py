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

import sys
from enum import Enum
import struct
import zipfile
import numpy as np
# conda install -c conda-forge commentjson
import commentjson as json
import tinycudann as tcnn
import torch

from .activation_functions import SnakeActivation, SnakeAltActivation


class SymmetrizerAdd(torch.nn.Module):
    def forward(self, t0, t1):
        return t0 + t1


class SymmetrizerAddDiff(torch.nn.Module):
    def forward(self, t0, t1):
        out = torch.cat([t0 + t1, torch.abs(t0 - t1)], dim=-1)
        return out


class SymmetrizerMul(torch.nn.Module):
    def forward(self, t0, t1):
        return t0 * t1


class NetworkType(Enum):
    TCNN = 0
    TCNN_SEPARATE = 1
    PYTORCH_MLP = 2
    TCNN_ENCODING_AND_PYTORCH_MLP = 3


def set_tcnn_network_weights(module, num_weights, weights_numpy):
    params = module.state_dict()['params']
    num_weights_expected = params.size()[0]
    if num_weights < num_weights_expected:
        raise Exception(
            f'Error in load_network_weights_tcnn: Mismatch in provided ({num_weights}) vs. expected ' +
            f'({num_weights_expected}) number of weights.')

    params_new = torch.tensor(weights_numpy[:num_weights_expected], device=torch.device('cuda'))
    with torch.no_grad():
        params[:] = params_new[:]
    return num_weights_expected


def padding_16(x):
    if x % 16 == 0:
        return x
    else:
        return x + 16 - (x % 16)


def set_pytorch_mlp_network_weights(module, num_weights, weights_numpy):
    num_weights_used = 0
    for layer_idx, layer in enumerate(module.network):
        if not hasattr(layer, 'weight'):
            continue
        size_2d = layer.weight.size()
        pad0 = padding_16(size_2d[0])
        pad1 = padding_16(size_2d[1])
        num_weights_padded = pad0 * pad1
        if num_weights_padded > num_weights:
            raise Exception(f'Error in set_pytorch_mlp_network_weights: Less weights were provided than expected.')
        weights_new = torch.tensor(weights_numpy[:num_weights_padded].reshape((pad0, pad1))[0:size_2d[0], 0:size_2d[1]])
        weights_numpy = weights_numpy[num_weights_padded:]
        num_weights -= num_weights_padded
        num_weights_used += num_weights_padded
        with torch.no_grad():
            layer.weight.copy_(weights_new.reshape(layer.weight.size()))
    return num_weights_used


def load_network_weights_tcnn(module, weights_file, network_type):
    header = struct.unpack('II', weights_file[0:8])
    format = header[0]
    if format != 0:
        raise Exception('Error in load_network_weights_tcnn: Currently, only float32 weights are supported.')
    num_weights = header[1]
    weights_numpy = np.frombuffer(weights_file[8:], dtype=np.float32)

    if network_type == NetworkType.TCNN:
        num_weights -= set_tcnn_network_weights(module, num_weights, weights_numpy)
    elif network_type == NetworkType.TCNN_SEPARATE:
        num_weights_network = set_tcnn_network_weights(module[1], num_weights, weights_numpy)
        weights_numpy = weights_numpy[num_weights_network:]
        num_weights -= num_weights_network
        # Set weights of input encoding.
        num_weights_encoding = set_tcnn_network_weights(module[0], num_weights, weights_numpy)
        # weights_numpy = weights_numpy[num_weights_encoding:]
        num_weights -= num_weights_encoding
    elif network_type == NetworkType.PYTORCH_MLP:
        num_weights -= set_pytorch_mlp_network_weights(module, num_weights, weights_numpy)
    elif network_type == NetworkType.TCNN_ENCODING_AND_PYTORCH_MLP:
        # Set weights of network.
        num_weights_network = set_pytorch_mlp_network_weights(module[1], num_weights, weights_numpy)
        weights_numpy = weights_numpy[num_weights_network:]
        num_weights -= num_weights_network
        # Set weights of input encoding.
        num_weights_encoding = set_tcnn_network_weights(module[0], num_weights, weights_numpy)
        # weights_numpy = weights_numpy[num_weights_encoding:]
        num_weights -= num_weights_encoding

    if num_weights != 0:
        raise Exception(
            f'Error in load_network_weights_tcnn: More weights were provided than expected (remaining: {num_weights}).')


def serialize_tcnn_network_weights(module):
    params = module.state_dict()['params']
    return params.cpu().numpy()


def serialize_pytorch_mlp_network_weights(module):
    weights_list = []
    for layer_idx, layer in enumerate(module.network):
        if not hasattr(layer, 'weight'):
            continue
        size_2d = layer.weight.size()
        pad0 = padding_16(size_2d[0])
        pad1 = padding_16(size_2d[1])
        weights_layer = np.zeros((pad0, pad1), dtype=np.float32)
        with torch.no_grad():
            weights_layer[:size_2d[0], :size_2d[1]] = layer.weight.cpu().numpy()[:size_2d[0], :size_2d[1]]
        weights_list.append(weights_layer.flatten())
    return np.concatenate(weights_list)


def serialize_network_weights_tcnn(module, network_type):
    weights_bytearray = bytearray()
    weights_bytearray.extend(int(0).to_bytes(4, sys.byteorder))

    weights = None
    if network_type == NetworkType.TCNN:
        weights = serialize_tcnn_network_weights(module)
    elif network_type == NetworkType.TCNN_SEPARATE:
        weights0 = serialize_tcnn_network_weights(module[1])
        weights1 = serialize_tcnn_network_weights(module[0])
        weights = np.concatenate((weights0, weights1))
    elif network_type == NetworkType.PYTORCH_MLP:
        weights = serialize_pytorch_mlp_network_weights(module)
    elif network_type == NetworkType.TCNN_ENCODING_AND_PYTORCH_MLP:
        weights0 = serialize_pytorch_mlp_network_weights(module[1])
        weights1 = serialize_tcnn_network_weights(module[0])
        weights = np.concatenate((weights0, weights1))

    weights_bytearray.extend(int(weights.shape[0]).to_bytes(4, sys.byteorder))
    weights_bytearray.extend(weights.tobytes())
    return weights_bytearray


def create_activation_layer(name):
    if name == 'ReLU':
        return torch.nn.ReLU()
    elif name == 'LeakyReLU':
        return torch.nn.LeakyReLU()
    elif name == 'Snake':
        return SnakeActivation()
    elif name == 'SnakeAlt':
        return SnakeAltActivation()
    else:
        raise Exception(f'Error in create_activation_layer: Invalid activation function name \'{name}\'.')


class PytorchMLP(torch.nn.Module):
    def __init__(self, num_input_dims, num_output_dims, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        num_neurons = config['n_neurons']
        num_hidden_layers = config['n_hidden_layers']
        for i in range(num_hidden_layers + 1):
            layer_input_dim = num_neurons
            layer_output_dim = num_neurons
            if i == 0:
                layer_input_dim = num_input_dims
            if i == num_hidden_layers:
                layer_output_dim = num_output_dims
            layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim, bias=False))
            if i != num_hidden_layers:
                layers.append(create_activation_layer(config['activation']))
            elif 'output_activation' in config and config['output_activation'] != 'None':
                layers.append(create_activation_layer(config['output_activation']))
        self.network = torch.nn.Sequential(*layers)
        self.num_input_dims = num_input_dims
        self.num_output_dims = num_output_dims
        self.num_neurons = num_neurons

    def forward(self, x):
        if x.dtype != torch.float:
            x = x.float()
        return self.network(x)


class CorrelationSRN(torch.nn.Module):
    def __init__(self, zip_filepath, use_tcnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        archive = zipfile.ZipFile(zip_filepath, 'r')
        self.config_data = archive.read('config.json')
        self.config_encoder_data = archive.read('config_encoder.json')
        self.config_decoder_data = archive.read('config_decoder.json')
        config = json.loads(self.config_data)
        config_encoder = json.loads(self.config_encoder_data)
        config_decoder = json.loads(self.config_decoder_data)

        num_input_dims_encoder = 3
        num_output_dims_encoder = config_encoder['network']['n_neurons']
        if config['symmetrizer_type'] == 'AddDiff':
            num_input_dims_decoder = num_output_dims_encoder * 2
        else:
            num_input_dims_decoder = num_output_dims_encoder
        num_output_dims_decoder = 1

        if use_tcnn:
            self.encoder = tcnn.NetworkWithInputEncoding(
                num_input_dims_encoder, num_output_dims_encoder,
                config_encoder['encoding'], config_encoder['network']
            )
            self.decoder = tcnn.NetworkWithInputEncoding(
                num_input_dims_decoder, num_output_dims_decoder,
                config_decoder['encoding'], config_decoder['network']
            )
            self.encoder_type = NetworkType.TCNN
            self.decoder_type = NetworkType.TCNN
        else:
            encoder_encoding = tcnn.Encoding(
                num_input_dims_encoder, config_encoder['encoding']
            )
            #encoder_network = tcnn.Network(
            #    encoder_encoding.n_output_dims, num_output_dims_encoder, config_encoder['network'])
            encoder_network = PytorchMLP(
                encoder_encoding.n_output_dims, num_output_dims_encoder, config_encoder['network'])
            self.encoder = torch.nn.Sequential(encoder_encoding, encoder_network)
            self.decoder = PytorchMLP(num_input_dims_decoder, num_output_dims_decoder, config_decoder['network'])
            #encoder_type = NetworkType.TCNN_SEPARATE
            self.encoder_type = NetworkType.TCNN_ENCODING_AND_PYTORCH_MLP
            self.decoder_type = NetworkType.PYTORCH_MLP
            encoder_network.to('cuda')
            self.decoder.to('cuda')

        if config['symmetrizer_type'] == 'Add':
            self.symmetrizer = SymmetrizerAdd()
        elif config['symmetrizer_type'] == 'AddDiff':
            self.symmetrizer = SymmetrizerAddDiff()
        elif config['symmetrizer_type'] == 'Mul':
            self.symmetrizer = SymmetrizerMul()

        if 'network_encoder.bin' in archive.namelist():
            weights_file = archive.read('network_encoder.bin')
            load_network_weights_tcnn(self.encoder, weights_file, self.encoder_type)
        if 'network_decoder.bin' in archive.namelist():
            weights_file = archive.read('network_decoder.bin')
            load_network_weights_tcnn(self.decoder, weights_file, self.decoder_type)

    def save_network(self, out_path):
        archive = zipfile.ZipFile(out_path, 'w')
        archive.writestr('config.json', self.config_data)
        archive.writestr('config_encoder.json', self.config_encoder_data)
        archive.writestr('config_decoder.json', self.config_decoder_data)

        weights_encoder = serialize_network_weights_tcnn(self.encoder, self.encoder_type)
        weights_decoder = serialize_network_weights_tcnn(self.decoder, self.decoder_type)
        archive.writestr('network_encoder.bin', weights_encoder)
        archive.writestr('network_decoder.bin', weights_decoder)

    def forward(self, pos0, pos1):
        enc0 = self.encoder(pos0)
        enc1 = self.encoder(pos1)
        sym = self.symmetrizer(enc0, enc1)
        output = self.decoder(sym)
        return output

    def apply_encoder(self, pos):
        return self.encoder(pos)

    def apply_decoder(self, enc0, enc1):
        sym = self.symmetrizer(enc0, enc1)
        output = self.decoder(sym)
        return output
