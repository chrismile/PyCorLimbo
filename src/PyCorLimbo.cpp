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
//<pybind11/functional.h>

#include <pybind11/functional.h>

#include "Correlation.hpp"
#include "MutualInformation.hpp"
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
    m.def("pearson_correlation", pearsonCorrelation,
          "Computes the Pearson correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("spearman_rank_correlation", spearmanRankCorrelation,
          "Computes the Spearman rank correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("kendall_rank_correlation", kendallRankCorrelation,
          "Computes the Kendall rank correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("mutual_information_binned", mutualInformationBinned,
          "Computes the mutual information of the Torch tensors X and Y using a binning estimator.",
          py::arg("X"), py::arg("Y"), py::arg("num_bins"),
          py::arg("X_min"), py::arg("X_max"), py::arg("Y_min"), py::arg("Y_max"));
    m.def("mutual_information_kraskov", mutualInformationKraskov,
          "Computes the mutual information of the Torch tensors X and Y using the Kraskov estimator.",
          py::arg("X"), py::arg("Y"), py::arg("k"));
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
    const int xs, ys, zs;
    const std::function<float(int, int, int, int, int, int)> f; //< Function to be optimized.
    mutable float bestValue = std::numeric_limits<float>::lowest();
    mutable std::array<int32_t, 6> bestSample;
    BO_PARAM(size_t, dim_in, 6);
    BO_PARAM(size_t, dim_out, 1);

    // Convert continuous to discrete indices with probabilistic reparameterization.
    Eigen::VectorXd operator()(const Eigen::VectorXd& v) const {
        int xi = pr(v[0] * (xs - 1));
        int yi = pr(v[1] * (ys - 1));
        int zi = pr(v[2] * (zs - 1));
        int xj = pr(v[3] * (xs - 1));
        int yj = pr(v[4] * (ys - 1));
        int zj = pr(v[5] * (zs - 1));
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
        throw std::runtime_error("Error in spearmanRankCorrelation: Sample tensor needs to have 2 dimensions.");
    }
    auto numSamples = int(sampleTensor.size(0));
    if (sampleTensor.size(1) != 6) {
        throw std::runtime_error("Error in spearmanRankCorrelation: Sample tensor dimension 1 needs to be 6.");
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
        auto eval = BayOpt::Eval{xs, ys, zs, callback};
        optimizer.optimize(eval);
        /*const auto& bestSample = optimizer.best_sample();
        for (int d = 0; d < 6; d++) {
            sampleAccessor[sampleIdx][d] = int(bestSample(d));
        }*/
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

void optimizeMultiThreaded(
        BayOptSettings settings, torch::Tensor sampleTensor,
        std::function<void(torch::Tensor, torch::Tensor)> callback) {
    if (sampleTensor.sizes().size() != 2) {
        throw std::runtime_error("Error in spearmanRankCorrelation: Sample tensor needs to have 2 dimensions.");
    }
    auto numSamples = int(sampleTensor.size(0));
    if (sampleTensor.size(1) != 6) {
        throw std::runtime_error("Error in spearmanRankCorrelation: Sample tensor dimension 1 needs to be 6.");
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
            auto eval = BayOpt::Eval{xs, ys, zs, callbackThread};
            optimizer.optimize(eval);
            /*const auto& bestSample = optimizer.best_sample();
            if (sampleIdx < numSamples) {
                for (int d = 0; d < 6; d++) {
                    sampleAccessor[sampleIdx][d] = int(bestSample(d));
                }
            }*/
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

torch::Tensor pearsonCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::PEARSON, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in pearsonCorrelation: Unsupported device.");
    }
}

torch::Tensor spearmanRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::SPEARMAN, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in spearmanRankCorrelation: Unsupported device.");
    }
}

torch::Tensor kendallRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::KENDALL, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in kendallRankCorrelation: Unsupported device.");
    }
}

torch::Tensor mutualInformationBinned(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t numBins,
        double referenceMin, double referenceMax, double queryMin, double queryMax) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_BINNED, int(numBins), 0,
                float(referenceMin), float(referenceMax), float(queryMin), float(queryMax));
    } else {
        throw std::runtime_error("Error in mutualInformationBinned: Unsupported device.");
    }
}

torch::Tensor mutualInformationKraskov(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV, 0, int(k),
                0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in mutualInformationKraskov: Unsupported device.");
    }
}

torch::Tensor computeCorrelationCpu(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, CorrelationMeasureType correlationMeasureType,
        int numBins, int k, float referenceMin, float referenceMax, float queryMin, float queryMax) {
    if (referenceTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in computeCorrelationCpu: referenceTensor.sizes().size() > 2.");
    }
    if (queryTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in computeCorrelationCpu: queryTensor.sizes().size() > 2.");
    }

    // Size of tensor: (M, N) or (N).
    const int64_t Mr = referenceTensor.sizes().size() == 1 ? 1 : referenceTensor.size(0);
    const int64_t Nr = referenceTensor.sizes().size() == 1 ? referenceTensor.size(0) : referenceTensor.size(1);
    const int64_t Mq = queryTensor.sizes().size() == 1 ? 1 : queryTensor.size(0);
    const int64_t Nq = queryTensor.sizes().size() == 1 ? queryTensor.size(0) : queryTensor.size(1);
    if (Nr != Nq || (Mr != Mq && Mr != 1 && Mq != 1)) {
        throw std::runtime_error("Error in mutualInformationKraskovCuda: Tensor size mismatch.");
    }
    const int64_t M = std::max(Mr, Mq);
    const int64_t N = Nr;

    torch::Tensor outputTensor = torch::zeros(M, at::TensorOptions().dtype(torch::kFloat32));
    auto referenceData = referenceTensor.data_ptr<float>();
    auto queryData = queryTensor.data_ptr<float>();
    auto outputAccessor = outputTensor.accessor<float, 1>();
    auto referenceStride = Mr == 1 ? 0 : uint32_t(referenceTensor.stride(0));
    auto queryStride = Mq == 1 ? 0 : uint32_t(queryTensor.stride(0));

    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computePearson2<float>(referenceValues, queryValues, int(N));
                outputAccessor[batchIdx] = miValue;
            }
        }
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            std::vector<std::pair<float, int>> ordinalRankArraySpearman;
            ordinalRankArraySpearman.reserve(N);
            auto* referenceRanks = new float[N];
            auto* queryRanks = new float[N];
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                computeRanks(referenceValues, referenceRanks, ordinalRankArraySpearman, int(N));
                computeRanks(queryValues, queryRanks, ordinalRankArraySpearman, int(N));
                float miValue = computePearson2<float>(referenceRanks, queryRanks, int(N));
                outputAccessor[batchIdx] = miValue;
            }
            delete[] referenceRanks;
            delete[] queryRanks;
        }
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(N);
            ordinalRankArray.reserve(N);
            y.reserve(N);
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computeKendall(referenceValues, queryValues, int(N), jointArray, ordinalRankArray, y);
                outputAccessor[batchIdx] = miValue;
            }
        }
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, numBins, referenceData, referenceStride, queryData, queryStride, outputAccessor) \
        shared(referenceMin, referenceMax, queryMin, queryMax)
#endif
        {
            auto* histogram0 = new float[numBins];
            auto* histogram1 = new float[numBins];
            auto* histogram2d = new float[numBins * numBins];
            auto* X = new float[N];
            auto* Y = new float[N];
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                for (int i = 0; i < N; i++) {
                    X[i] = (referenceValues[i] - referenceMin) / (referenceMax - referenceMin);
                    Y[i] = (queryValues[i] - queryMin) / (queryMax - queryMin);
                }
                float miValue = computeMutualInformationBinned<float>(
                        X, Y, int(numBins), int(N), histogram0, histogram1, histogram2d);
                outputAccessor[batchIdx] = miValue;
            }
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
            delete[] X;
            delete[] Y;
        }
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, k, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            KraskovEstimatorCache<float> kraskovEstimatorCache;
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computeMutualInformationKraskov<float>(
                        referenceValues, queryValues, int(k), int(N), kraskovEstimatorCache);
                outputAccessor[batchIdx] = miValue;
            }
        }
    }

    return outputTensor;
}
