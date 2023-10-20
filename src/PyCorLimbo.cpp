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

#define USE_NLOPT
#include <limbo/bayes_opt/boptimizer.hpp>
#include <utility>
#include <pybind11/functional.h>

#include "PyCorLimbo.hpp"

PYBIND11_MODULE(pycorlimbo, m) {
    py::class_<BayOptSettings>(m, "BayOptSettings")
            .def(py::init<>())
            .def_readwrite("xs", &BayOptSettings::xs)
            .def_readwrite("ys", &BayOptSettings::ys)
            .def_readwrite("zs", &BayOptSettings::zs)
            .def_readwrite("num_init_samples", &BayOptSettings::num_init_samples)
            .def_readwrite("num_iterations", &BayOptSettings::num_iterations)
            .def_readwrite("num_optimizer_iterations", &BayOptSettings::num_optimizer_iterations)
            .def_readwrite("alpha", &BayOptSettings::alpha);
    m.def("optimize_single_threaded", optimizeSingleThreaded,
          "Applies Bayesian optimization (single-threaded).",
          py::arg("settings"), py::arg("sample_tensor"), py::arg("callback"));
    m.def("optimize_multi_threaded", optimizeMultiThreaded,
          "Applies Bayesian optimization (multi-threaded).",
          py::arg("settings"), py::arg("sample_tensor"), py::arg("callback"));
    m.def("optimize_multi_threaded_blocks", optimizeMultiThreadedBlocks,
          "Applies Bayesian optimization (multi-threaded).",
          py::arg("settings"), py::arg("sample_tensor"), py::arg("block_size"), py::arg("block_offsets"),
          py::arg("callback"));
}

namespace BayOpt {

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, false);
    };
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };
    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };
    struct init_randomsampling {
        BO_DYN_PARAM(int, samples);
    };
    struct stop_maxiterations {
        BO_DYN_PARAM(int, iterations);
    };
    struct acqui_ucb : public defaults::acqui_ucb {
        BO_DYN_PARAM(double, alpha);
    };
};

BO_DECLARE_DYN_PARAM(int, BayOpt::Params::stop_maxiterations, iterations);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::init_randomsampling, samples);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(double, BayOpt::Params::acqui_ucb, alpha);

std::random_device rand_dev;
std::mt19937 gen(rand_dev());
std::uniform_real_distribution<> uniform_dist;

inline int pr(double x){
    double base = std::floor(x);
    return int(base) + int((x - base) > uniform_dist(gen));
}

struct Eval {
    const int oxi, oyi, ozi, oxj, oyj, ozj;
    const int xs, ys, zs;
    const std::function<float(int, int, int, int, int, int)> f; //< Function to be optimized.
    mutable float bestValue = std::numeric_limits<float>::lowest();
    mutable std::array<int32_t, 6> bestSample;
    BO_PARAM(size_t, dim_in, 6);
    BO_PARAM(size_t, dim_out, 1);

    // Convert continuous to discrete indices with probabilistic reparameterization.
    Eigen::VectorXd operator()(const Eigen::VectorXd& v) const {
        int xi = oxi + pr(v[0] * (xs - 1));
        int yi = oyi + pr(v[1] * (ys - 1));
        int zi = ozi + pr(v[2] * (zs - 1));
        int xj = oxj + pr(v[3] * (xs - 1));
        int yj = oyj + pr(v[4] * (ys - 1));
        int zj = ozj + pr(v[5] * (zs - 1));
        float value = f(xi, yi, zi, xj, yj, zj);
        if (std::isnan(value)) {
            std::cerr << "Error: NaN sample detected." << std::endl;
            value = 0.0f;
        }
        if (value > bestValue) {
            bestValue = value;
            bestSample[0] = xi;
            bestSample[1] = yi;
            bestSample[2] = zi;
            bestSample[3] = xj;
            bestSample[4] = yj;
            bestSample[5] = zj;
        }
        return tools::make_vector(value);
    }
};

}

void optimizeSingleThreaded(
        BayOptSettings settings, torch::Tensor sampleTensor,
        std::function<float(int, int, int, int, int, int)> callback) {
    if (sampleTensor.sizes().size() != 2) {
        throw std::runtime_error("Error in optimizeMultiThreadedBlocks: Sample tensor needs to have 2 dimensions.");
    }
    auto numSamples = int(sampleTensor.size(0));
    if (sampleTensor.size(1) != 6) {
        throw std::runtime_error("Error in optimizeMultiThreadedBlocks: Sample tensor dimension 1 needs to be 6.");
    }
    auto sampleAccessor = sampleTensor.accessor<int32_t, 2>();

    int xs = settings.xs;
    int ys = settings.ys;
    int zs = settings.zs;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
        limbo::bayes_opt::BOptimizer<BayOpt::Params> optimizer;
        BayOpt::Params::stop_maxiterations::set_iterations(settings.num_iterations);
        BayOpt::Params::init_randomsampling::set_samples(settings.num_init_samples);
        BayOpt::Params::opt_nloptnograd::set_iterations(settings.num_optimizer_iterations);
        BayOpt::Params::acqui_ucb::set_alpha(settings.alpha);
        auto eval = BayOpt::Eval{0, 0, 0, 0, 0, 0, xs, ys, zs, callback};
        optimizer.optimize(eval);
        for (int d = 0; d < 6; d++) {
            sampleAccessor[sampleIdx][d] = eval.bestSample[d];
        }
    }
}

inline int iceil(int x, int y) { return (x - 1) / y + 1; }

class ThreadBarrier {
public:
    explicit ThreadBarrier(int numThreads) : numThreads(numThreads), conditionCounter(0), waitingCounter(0) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutexObj);
        ++conditionCounter;
        ++waitingCounter;
        conditionVariable.wait(lock, [&]{return conditionCounter >= numThreads;});
        conditionVariable.notify_one();
        --waitingCounter;
        if(waitingCounter == 0) {
            conditionCounter = 0;
        }
        lock.unlock();
    }

private:
    std::mutex mutexObj;
    std::condition_variable conditionVariable;
    int numThreads;
    int conditionCounter;
    int waitingCounter;
};

#define CLAMP_BLOCK(offset, block, size) \
    if (offset + block > size) { \
        block = size - offset; \
    }

void optimizeMultiThreadedBlocks(
        BayOptSettings settings, torch::Tensor sampleTensor, int blockSize, torch::Tensor blockOffsets,
        std::function<void(torch::Tensor, torch::Tensor)> callback) {
    if (sampleTensor.sizes().size() != 2) {
        throw std::runtime_error("Error in optimizeMultiThreadedBlocks: Sample tensor needs to have 2 dimensions.");
    }
    auto numSamples = int(sampleTensor.size(0));
    if (sampleTensor.size(1) != 6) {
        throw std::runtime_error("Error in optimizeMultiThreadedBlocks: Sample tensor dimension 1 needs to be 6.");
    }
    if (blockSize > 0) {
        if (blockOffsets.size(1) != 6) {
            throw std::runtime_error(
                    "Error in optimizeMultiThreadedBlocks: Block offset tensor dimension 1 needs to be 6.");
        }
    }
    auto sampleAccessor = sampleTensor.accessor<int32_t, 2>();

    int xs = settings.xs;
    int ys = settings.ys;
    int zs = settings.zs;

    auto numThreads = int(std::thread::hardware_concurrency());
    std::vector<std::thread> threads(numThreads);
    torch::Tensor queryTensor = torch::zeros({int(threads.size()), 6}, torch::TensorOptions().dtype(torch::kInt32));
    auto queryAccessor = queryTensor.accessor<int32_t, 2>();
    torch::Tensor resultsTensor = torch::zeros({int(threads.size())}, torch::TensorOptions().dtype(torch::kFloat32));
    auto resultsAccessor = resultsTensor.accessor<float, 1>();
    std::atomic<int> freeSampleIdx{};

    ThreadBarrier queryBarrier(numThreads + 1);
    ThreadBarrier resultsBarrier(numThreads + 1);

    const int numSamplesPadded = iceil(numSamples, numThreads) * numThreads;
    auto threadFunc = [&](int threadIdx) {
        auto callbackThread = [&](int xi, int yi, int zi, int xj, int yj, int zj) {
            queryAccessor[threadIdx][0] = xi;
            queryAccessor[threadIdx][1] = yi;
            queryAccessor[threadIdx][2] = zi;
            queryAccessor[threadIdx][3] = xj;
            queryAccessor[threadIdx][4] = yj;
            queryAccessor[threadIdx][5] = zj;

            queryBarrier.wait();
            resultsBarrier.wait();
            float result = resultsAccessor[threadIdx];

            return result;
        };

        while (true) {
            int sampleIdx = freeSampleIdx++;
            if (sampleIdx >= numSamplesPadded) {
                break;
            }
            limbo::bayes_opt::BOptimizer<BayOpt::Params> optimizer;
            BayOpt::Params::stop_maxiterations::set_iterations(settings.num_iterations);
            BayOpt::Params::init_randomsampling::set_samples(settings.num_init_samples);
            BayOpt::Params::opt_nloptnograd::set_iterations(settings.num_optimizer_iterations);
            BayOpt::Params::acqui_ucb::set_alpha(settings.alpha);
            int oxi = 0, oyi = 0, ozi = 0, oxj = 0, oyj = 0, ozj = 0;
            int bxs = xs, bys = ys, bzs = zs;
            if (blockSize > 0) {
                bxs = blockSize;
                bys = blockSize;
                bzs = blockSize;
                auto blockOffsetsAccessor = queryTensor.accessor<int32_t, 2>();
                oxi = sampleAccessor[sampleIdx][0];
                oyi = sampleAccessor[sampleIdx][1];
                ozi = sampleAccessor[sampleIdx][2];
                oxj = sampleAccessor[sampleIdx][3];
                oyj = sampleAccessor[sampleIdx][4];
                ozj = sampleAccessor[sampleIdx][5];
                CLAMP_BLOCK(oxi, bxs, xs);
                CLAMP_BLOCK(oyi, bys, ys);
                CLAMP_BLOCK(ozi, bzs, zs);
                CLAMP_BLOCK(oxj, bxs, xs);
                CLAMP_BLOCK(oyj, bys, ys);
                CLAMP_BLOCK(ozj, bzs, zs);
            }
            auto eval = BayOpt::Eval{oxi, oyi, ozi, oxj, oyj, ozj, bxs, bys, bzs, callbackThread};
            optimizer.optimize(eval);
            if (sampleIdx < numSamples) {
                for (int d = 0; d < 6; d++) {
                    sampleAccessor[sampleIdx][d] = eval.bestSample[d];
                }
            }
        }
    };

    for (int threadIdx = 0; threadIdx < int(threads.size()); threadIdx++) {
        threads[threadIdx] = std::thread(threadFunc, threadIdx);
    }
    int iteration = 0;
    const int numSamplesTotal = settings.num_init_samples + settings.num_iterations;
    const int maxIteration = iceil(numSamples, int(threads.size())) * numSamplesTotal;
    while (iteration < maxIteration) {
        queryBarrier.wait();
        callback(queryTensor, resultsTensor);
        resultsBarrier.wait();
        iteration++;
    }
    for (auto& t : threads) {
        t.join();
    }
}

void optimizeMultiThreaded(
        BayOptSettings settings, torch::Tensor sampleTensor,
        std::function<void(torch::Tensor, torch::Tensor)> callback) {
    optimizeMultiThreadedBlocks(settings, std::move(sampleTensor), 0, torch::Tensor{}, std::move(callback));
}
