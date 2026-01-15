#pragma once
#include <onnxruntime_cxx_api.h>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

class BotONNX {
public:
    BotONNX(const std::string& onnxPath, int obsDim, int actDim)
    : env(ORT_LOGGING_LEVEL_WARNING, "arcomage"),
      obsDim(obsDim), actDim(actDim),
      rng(std::random_device{}())
    {
        Ort::SessionOptions opt;
        opt.SetIntraOpNumThreads(1);
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, onnxPath.c_str(), opt);

        allocator = Ort::AllocatorWithDefaultOptions();
        inputName  = session.GetInputNameAllocated(0, allocator).get();
        outputName = session.GetOutputNameAllocated(0, allocator).get();
    }

    int act(const std::vector<float>& obs, const std::vector<uint8_t>& mask, bool stochastic=true) {
        std::vector<int64_t> shape{1, (int64_t)obsDim};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value in = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(obs.data()), obs.size(),
            shape.data(), shape.size()
        );

        const char* inNames[] = { inputName.c_str() };
        const char* outNames[] = { outputName.c_str() };

        auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &in, 1, outNames, 1);
        float* logits = outs[0].GetTensorMutableData<float>();

        std::vector<double> probs(actDim, 0.0);
        double maxLogit = -1e100;
        for(int i=0;i<actDim;i++) if(mask[i]) maxLogit = std::max(maxLogit, (double)logits[i]);

        double sum = 0.0;
        for(int i=0;i<actDim;i++){
            if(!mask[i]) continue;
            double x = std::exp((double)logits[i] - maxLogit);
            probs[i] = x;
            sum += x;
        }

        if(sum <= 0.0){
            for(int i=0;i<actDim;i++) if(mask[i]) return i;
            return 0;
        }

        for(double& p : probs) p /= sum;

        if(!stochastic){
            return (int)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputName, outputName;
    int obsDim, actDim;
    std::mt19937 rng;
};
