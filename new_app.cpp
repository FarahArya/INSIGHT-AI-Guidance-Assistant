#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

using json = nlohmann::json;

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

        // 🔁 Call Python script to get AI detection result
        FILE *pipe = popen("python3 Insight/insight_deploy/insight_infer.py", "r");
        if (!pipe)
        {
            std::cerr << "[ERROR] Failed to run Python script." << std::endl;
            continue;
        }

        char buffer[256];
        if (!fgets(buffer, sizeof(buffer), pipe))
        {
            std::cerr << "[ERROR] No output from Python script." << std::endl;
            pclose(pipe);
            continue;
        }
        pclose(pipe);

        std::string output_json(buffer);
        json j;
        try
        {
            j = json::parse(output_json);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Failed to parse JSON: " << output_json << std::endl;
            continue;
        }

        if (!j.contains("text"))
        {
            std::cerr << "[ERROR] JSON has no 'text' field." << std::endl;
            continue;
        }

        std::string sentence = j["text"];
        std::ofstream out("say.json");
        out << j.dump() << std::endl;
        out.close();

        std::cout << "[INFO] Sending text to Piper: " << sentence << std::endl;

        std::string cmd = "cat say.json | ./piper/piper "
                          "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                          "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                          "--output_file spoken.wav "
                          "--json-input && aplay spoken.wav";

        std::system(cmd.c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}
