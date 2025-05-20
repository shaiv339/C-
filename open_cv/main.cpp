// Project 1: Real-Time Object Detection with OpenCV (Embedded System)
// File: main.cpp
// Dependencies: OpenCV (DNN module)
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // Load YOLOv4 model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov4.cfg", "yolov4.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  // On ARM, this will use NEON optimizations if available

    cv::VideoCapture cap(0);  // Open default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocess frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(), true, false);
        net.setInput(blob);

        // Run inference
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Postprocess detections (simplified)
        float confThreshold = 0.5;
        for (auto &output : outputs) {
            auto data = (float*)output.data;
            for (int i = 0; i < output.total(); i += 85) {
                float score = data[4];
                if (score > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width   = (int)(data[2] * frame.cols);
                    int height  = (int)(data[3] * frame.rows);
                    int left    = centerX - width / 2;
                    int top     = centerY - height / 2;
                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                }
                data += 85;
            }
        }

        // Display
        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) == 27) break;  // Exit on ESC key
    }
    return 0;
}