#include "detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    Detector detector("assets/yolov5s.onnx");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera.\n";
        return -1;
    }

    std::cout << "Starting real-time detection...\n";
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        detector.detect(frame);
        cv::imshow("YOLOv5 Real-Time Detection", frame);
        if (cv::waitKey(1) == 27) break;  // ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}