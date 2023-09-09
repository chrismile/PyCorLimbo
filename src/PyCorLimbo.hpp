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

#include "CorrelationDefines.hpp"

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

torch::Tensor pearsonCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor);
torch::Tensor spearmanRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor);
torch::Tensor kendallRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor);
torch::Tensor mutualInformationBinned(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t numBins,
        double referenceMin, double referenceMax, double queryMin, double queryMax);
torch::Tensor mutualInformationKraskov(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k);

// numBins (MI binned) and k (MI Kraskov) are zero when not needed by the correlation measure type.
torch::Tensor computeCorrelationCpu(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, CorrelationMeasureType correlationMeasureType,
        int numBins, int k, float referenceMin, float referenceMax, float queryMin, float queryMax);

#endif //PYCORIANDER_PYCORIANDER_HPP
