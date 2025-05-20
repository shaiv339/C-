#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Detector {
public:
    Detector(const std::string& model_path);
    void detect(cv::Mat& frame);

private:
    cv::dnn::Net net_;
    float conf_threshold_ = 0.5;
    float nms_threshold_ = 0.4;
    std::vector<std::string> class_names_;
    void load_class_names();
};