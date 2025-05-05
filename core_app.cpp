#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>

using json = nlohmann::json;

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;

std::vector<float> preprocessFrame(const cv::Mat &frame)
{
    cv::Mat resized, rgb, float_img;
    std::cout << "[INFO] Preprocessing frame..." << std::endl;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < float_img.rows; ++i)
            for (int j = 0; j < float_img.cols; ++j)
                tensor.push_back(float_img.at<cv::Vec3f>(i, j)[c]);

    return tensor;
}

int main()
{
    std::cout << "[INFO] Starting AI Guidance Assistant..." << std::endl;

    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Could not open camera." << std::endl;
        return 1;
    }
    std::cout << "[INFO] Camera opened successfully." << std::endl;

    // Load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InsightAI");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "yolo11n.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char *input_name = input_name_ptr.get();
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char *output_name = output_name_ptr.get();

    std::cout << "[INFO] ONNX model loaded successfully." << std::endl;

    while (true)
    {
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "[INFO] Capturing frame..." << std::endl;

        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "[WARN] Frame capture failed. Skipping..." << std::endl;
            continue;
        }
        std::cout << "[INFO] Frame captured." << std::endl;

        std::vector<float> input_tensor_values = preprocessFrame(frame);
        std::vector<int64_t> input_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size());

        std::cout << "[INFO] Running model inference..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
            &output_name, 1);
        std::cout << "[INFO] Inference complete." << std::endl;

        float *output = output_tensors[0].GetTensorMutableData<float>();

        // ⚠️ Replace with real parsing of model output
        std::string label = "object";
        int distance = 2;

        json j;
        j["text"] = "There is a " + label + " approximately " + std::to_string(distance) + " metres ahead.";
        std::ofstream out("say.json");
        out << j.dump() << std::endl;
        out.close();

        std::cout << "[INFO] Sending text to Piper: " << j["text"] << std::endl;

        std::string cmd = "cat say.json | ./piper "
                          "--model voices/en_US-amy-medium/en_US-amy-medium.onnx "
                          "--config voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                          "--json-input";
        std::system(cmd.c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
