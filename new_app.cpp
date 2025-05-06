#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>

using json = nlohmann::json;

int main()
{
    std::cout << "[INFO] Starting AI Guidance Assistant..." << std::endl;

    // Just to keep the camera active and warmed up
    // cv::VideoCapture cap(0);
    // if (!cap.isOpened())
    // {
    //     std::cerr << "[ERROR] Could not open camera." << std::endl;
    //     return 1;
    // }
    // std::cout << "[INFO] Camera opened successfully." << std::endl;

    // Start Python script via popen
    std::cout << "[INFO] Starting Python inference script..." << std::endl;
    FILE *pipe = popen(".venv/bin/python Insight/insight_deploy/insight_infer.py", "r");
    if (!pipe)
    {
        std::cerr << "[ERROR] Failed to run Python script." << std::endl;
        return 1;
    }

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe))
    {
        std::string line(buffer);
        // Skip lines that are not JSON
        if (line.find("{\"text\"") == std::string::npos)
            continue;

        try
        {
            json j = json::parse(line);
            std::string text = j["text"];
            std::cout << "[INFO] Sending to Piper: " << text << std::endl;

            // Save JSON to file for Piper
            std::ofstream out("say.json");
            out << j.dump() << std::endl;
            out.close();

            std::string cmd =
                "cat say.json | ./piper/piper "
                "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                "--output_file spoken.wav "
                "--json-input && aplay spoken.wav";

            std::system(cmd.c_str());
        }
        catch (std::exception &e)
        {
            std::cerr << "[ERROR] JSON parse error: " << e.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    pclose(pipe);
    return 0;
}
