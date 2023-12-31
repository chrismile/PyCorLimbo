/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PYCORIANDER_PYCORIANDER_HPP
#define PYCORIANDER_PYCORIANDER_HPP

#include <torch/script.h>
#include <torch/types.h>
#include <torch/extension.h>

struct BayOptSettings {
    int xs = 0;
    int ys = 0;
    int zs = 0;
    int num_init_samples = 10;
    int num_iterations = 10;
    int num_optimizer_iterations = 100;
    float alpha = 0.5f;
};

void optimizeSingleThreaded(
        BayOptSettings settings, torch::Tensor sampleTensor,
        std::function<float(int, int, int, int, int, int)> callback);
void optimizeMultiThreaded(
        BayOptSettings settings, torch::Tensor sampleTensor,
        std::function<void(torch::Tensor, torch::Tensor)> callback);
void optimizeMultiThreadedBlocks(
        BayOptSettings settings, torch::Tensor sampleTensor, int blockSize, torch::Tensor blockOffsets,
        std::function<void(torch::Tensor, torch::Tensor)> callback);

#endif //PYCORIANDER_PYCORIANDER_HPP
