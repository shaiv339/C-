#include "llama_wrapper.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    LlamaWrapper llama("../llama.cpp/models/ggml-model-q4_0.bin"); // Adjust path

    llama.inferAsync("The future of AI is", 50, [](const std::string& token) {
        std::cout << token << std::flush;
    });

    // Simulate app running while inference is async
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}