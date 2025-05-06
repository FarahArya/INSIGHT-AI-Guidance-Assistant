#include <iostream>
#include <fstream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>
#include <cstdio>

using json = nlohmann::json;

int main()
{
    std::cout << "[INFO] Starting AI Guidance Assistant (Trigger-based)...\n";

    while (true)
    {
        std::cout << "------------------------------------------\n";
        std::cout << "[INFO] Capturing frame...\n";

        // Create trigger file to signal the Python script to start
        std::ofstream trigger("trigger.txt");
        trigger << "1";
        trigger.close();

        // Run Python script and capture its output
        std::string command = ".venv/bin/python3 Insight/insight_deploy/insight_infer.py";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            std::cerr << "[ERROR] Failed to start Python script.\n";
            continue;
        }

        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            std::string jsonStr(buffer);
            try
            {
                json j = json::parse(jsonStr);
                std::string text = j["text"];
                std::cout << "[INFO] Got from Python: " << text << "\n";

                std::ofstream out("say.json");
                out << j.dump();
                out.close();

                std::string tts = "cat say.json | ./piper/piper "
                                  "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                                  "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                                  "--output_file spoken.wav "
                                  "--json-input && aplay spoken.wav";
                std::system(tts.c_str());
            }
            catch (...)
            {
                std::cerr << "[ERROR] Failed to parse Python output.\n";
            }
        }
        else
        {
            std::cerr << "[ERROR] No output from Python script.\n";
        }

        pclose(pipe);

        // Sleep a bit to simulate next iteration trigger pacing
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    return 0;
}
