#pragma once

#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <memory>
#include "llama.h"  // From llama.cpp

class LlamaWrapper {
public:
    explicit LlamaWrapper(const std::string& model_path);
    ~LlamaWrapper();

    void inferAsync(const std::string& prompt,
                    int max_tokens,
                    std::function<void(const std::string&)> callback);

private:
    llama_context* ctx_;
    llama_context_params params_;
    std::thread worker_;
    std::atomic<bool> running_;
};