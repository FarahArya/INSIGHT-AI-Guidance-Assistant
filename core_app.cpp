#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>

std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

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

        // Getting the YoloV Style
        // auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // std::cout << "[DEBUG] Output shape: ";
        // for (auto s : output_shape)
        //     std::cout << s << " ";
        // std::cout << std::endl;
        /*--------------------------------------AI Model---------------------------------------------------------*/
        float *output = output_tensors[0].GetTensorMutableData<float>();
        const float conf_threshold = 0.4f;

        int best_idx = -1;
        float best_conf = 0.0f;
        int best_class = -1;

        for (int i = 0; i < 8400; ++i)
        {
            float x = output[i * 84 + 0];
            float y = output[i * 84 + 1];
            float w = output[i * 84 + 2];
            float h = output[i * 84 + 3];
            float objectness = output[i * 84 + 4];

            float final_conf = objectness * max_class_score;
            if (final_conf < 0.5f)
                continue;

            // Find top class score
            int class_id = -1;
            float max_class_score = 0.0f;
            for (int c = 0; c < 80; ++c)
            {
                float class_score = output[i * 84 + 5 + c];
                if (class_score > max_class_score)
                {
                    max_class_score = class_score;
                    class_id = c;
                }
            }

            float final_conf = objectness * max_class_score;
            if (final_conf > best_conf)
            {
                best_conf = final_conf;
                best_idx = i;
                best_class = class_id;
            }
        }

        std::string label = "object";
        int distance = 2;

        if (best_idx >= 0 && best_class >= 0)
        {
            label = COCO_LABELS[best_class];

            float y = output[best_idx * 84 + 1];
            float h = output[best_idx * 84 + 3];

            // Estimate distance based on bbox height in pixels
            float box_height_px = h * INPUT_HEIGHT;
            float real_height_m = 1.7; // assume 1.7m if person
            if (label == "car")
                real_height_m = 1.5;
            if (label == "dog")
                real_height_m = 0.5;

            const float focal_px = 600.0f;
            distance = static_cast<int>((real_height_m * focal_px) / box_height_px);

            if (distance <= 0 || distance > 15)
            {
                std::cout << "[INFO] Skipping object: unrealistic distance\n";
                continue; // skip this detection
            }
        }
        /*--------------------------------------End of AI Model---------------------------------------------------------*/

        json j;
        j["text"] = "There is a " + label + " approximately " + std::to_string(distance) + " metres ahead.";
        std::ofstream out("say.json");
        out << j.dump() << std::endl;
        out.close();

        std::cout << "[INFO] Sending text to Piper: " << j["text"] << std::endl;

        std::string cmd = "cat say.json | ./piper/piper "
                          "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                          "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                          "--output_file spoken.wav "
                          "--json-input && aplay spoken.wav";

        std::system(cmd.c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
