#include "detector.h"
#include <fstream>

Detector::Detector(const std::string& model_path) {
    net_ = cv::dnn::readNetFromONNX(model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    load_class_names();
}

void Detector::load_class_names() {
    class_names_ = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        // Add more if needed from COCO
    };
}

void Detector::detect(cv::Mat& frame) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, {640, 640}, cv::Scalar(), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = frame.cols / 640.0f;
    float y_factor = frame.rows / 640.0f;

    for (const auto& output : outputs) {
        float* data = reinterpret_cast<float*>(output.data);
        for (int i = 0; i < output.rows; ++i, data += output.cols) {
            float conf = data[4];
            if (conf < conf_threshold_) continue;
            float* class_scores = data + 5;
            cv::Mat scores(1, class_names_.size(), CV_32FC1, class_scores);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > conf_threshold_) {
                int cx = static_cast<int>(data[0] * x_factor);
                int cy = static_cast<int>(data[1] * y_factor);
                int w = static_cast<int>(data[2] * x_factor);
                int h = static_cast<int>(data[3] * y_factor);
                int left = cx - w / 2;
                int top = cy - h / 2;

                class_ids.push_back(class_id_point.x);
                confidences.push_back(static_cast<float>(max_class_score));
                boxes.emplace_back(left, top, w, h);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    for (int i : indices) {
        cv::Rect box = boxes[i];
        int class_id = class_ids[i];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = class_names_[class_id] + ": " + cv::format("%.2f", confidences[i]);
        cv::putText(frame, label, {box.x, box.y - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
}