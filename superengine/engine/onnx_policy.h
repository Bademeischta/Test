#pragma once
#include <array>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& model);
    std::pair<std::array<float, 64>, float>
    operator()(const std::array<float, 18 * 8 * 8>& features);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
