// Project 2: Multithreaded LLM Inference Wrapper in C++ 
// File: llama_wrapper.h
#pragma once

#include <string>
#include <functional>
#include <thread>
#include "llama.h"  // llama.cpp C API header

class LlamaWrapper {
public:
    LlamaWrapper(const std::string &model_path) {
        params_ = llama_context_default_params();
        ctx_ = llama_init_from_file(model_path.c_str(), params_);
    }
    ~LlamaWrapper() {
        llama_free(ctx_);
    }

    // Runs inference asynchronously and pushes partial results via callback
    void inferAsync(const std::string &prompt,
                    int max_tokens,
                    std::function<void(const std::string &)> callback) {
        std::thread([this, prompt, max_tokens, callback]() {
            llama_tokenize(ctx_, prompt.c_str(), false);
            std::string output;
            for (int i = 0; i < max_tokens; ++i) {
                llama_eval(ctx_, 1, llama_token_eos());
                int token = llama_sample_top_p(ctx_, 1, 0.95f);
                const char *tok_str = llama_token_to_str(ctx_, token);
                output += tok_str;
                callback(output);
            }
        }).detach();
    }

private:
    llama_context *ctx_;
    llama_context_params params_;
};

// Usage example (main.cpp)
//#include "llama_wrapper.h"
//
// int main() {
//     LlamaWrapper wrapper("models/ggml-model.bin");
//     wrapper.inferAsync("Hello, world!", 50, [](const std::string &s) {
//         std::cout << s << std::flush;
//     });
//     std::this_thread::sleep_for(std::chrono::seconds(2));
//     return 0;