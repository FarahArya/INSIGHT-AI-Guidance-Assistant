#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Absolute paths to match Python script
// Update paths to match container mounted volume
const std::string TRIGGER_PATH = "./shared/trigger.txt";
const std::string FEEDBACK_PATH = "./shared/feedback.json";

void triggerPythonScript()
{
    std::ofstream triggerFile(TRIGGER_PATH);
    triggerFile << "run" << std::endl;
    triggerFile.close();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

bool waitForResponse(std::string &detectedText)
{
    const int maxRetries = 20;
    int attempts = 0;

    while (attempts < maxRetries)
    {
        std::ifstream responseFile(FEEDBACK_PATH);
        if (responseFile.good())
        {
            json j;
            responseFile >> j;
            detectedText = j.value("text", "");
            responseFile.close();
            std::remove(FEEDBACK_PATH.c_str()); // Clean up
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        attempts++;
    }
    return false;
}

int main()
{
    std::cout << "[INFO] AI Guidance Assistant started.\n";

    while (true)
    {
        triggerPythonScript();
        std::cout << "[INFO] Triggered Python script to capture and process frame.\n";

        std::string detectedText;
        if (waitForResponse(detectedText))
        {
            std::cout << "[INFO] Received response: " << detectedText << "\n";

            // Create JSON string and pipe directly to Piper
            json jsonOutput = {{"text", detectedText}};
            std::string jsonStr = jsonOutput.dump();

            std::string cmd = "echo '" + jsonStr + "' | ./piper/piper "
                                                   "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                                                   "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                                                   "--output_file spoken.wav "
                                                   "--json-input && aplay spoken.wav";

            std::system(cmd.c_str());
        }
        else
        {
            std::cerr << "[ERROR] No response from Python script.\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}