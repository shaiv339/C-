#include "llama_wrapper.h"
#include <iostream>
#include <sstream>

LlamaWrapper::LlamaWrapper(const std::string& model_path) {
    params_ = llama_context_default_params();
    ctx_ = llama_init_from_file(model_path.c_str(), params_);
    running_ = false;
}

LlamaWrapper::~LlamaWrapper() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    llama_free(ctx_);
}

void LlamaWrapper::inferAsync(const std::string& prompt,
                               int max_tokens,
                               std::function<void(const std::string&)> callback) {
    if (running_) return;  // prevent multiple inferences
    running_ = true;

    worker_ = std::thread([=]() {
        std::ostringstream output;
        std::vector<llama_token> tokens(prompt.size() + 32);
        int n = llama_tokenize(ctx_, prompt.c_str(), tokens.data(), tokens.size(), true);
        tokens.resize(n);

        llama_eval(ctx_, tokens.data(), tokens.size(), 0, 1);

        for (int i = 0; i < max_tokens && running_; ++i) {
            int token = llama_sample_top_p(ctx_, nullptr, 0.95f, 0.9f, 0.0f);
            llama_eval(ctx_, &token, 1, tokens.size() + i, 1);
            const char* token_str = llama_token_to_str(ctx_, token);
            output << token_str;
            callback(token_str);
            if (token == llama_token_eos()) break;
        }

        running_ = false;
    });
}