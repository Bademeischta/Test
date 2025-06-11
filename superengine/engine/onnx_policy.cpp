#include "onnx_policy.h"

OnnxPolicy::OnnxPolicy(const std::string& model)
    : env_(ORT_LOGGING_LEVEL_WARNING, "super"),
      session_(env_, model.c_str(), Ort::SessionOptions{nullptr}) {
    input_names_.push_back(session_.GetInputNameAllocated(0, alloc_));
    output_names_.push_back(session_.GetOutputNameAllocated(0, alloc_));
    output_names_.push_back(session_.GetOutputNameAllocated(1, alloc_));
}

std::pair<std::array<float, 64>, float>
OnnxPolicy::operator()(const std::array<float, 18 * 8 * 8>& feat) {
    static Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const int64_t shape[4] = {1, 18, 8, 8};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(feat.data()), feat.size(), shape, 4);

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(),
                                &input, 1, output_names_.data(), 2);

    auto* p_data = outputs[0].GetTensorMutableData<float>();
    auto* v_data = outputs[1].GetTensorMutableData<float>();

    std::array<float, 64> policy;
    std::copy(p_data, p_data + 64, policy.begin());
    float value = v_data[0];
    return {policy, value};
}
