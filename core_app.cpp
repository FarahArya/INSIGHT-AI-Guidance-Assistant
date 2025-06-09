#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ------------------------------------------------------------------
// paths shared with the Python container/script
const std::string TRIGGER_PATH = "./shared/trigger.txt";
const std::string FEEDBACK_PATH = "./shared/feedback.json";
// ------------------------------------------------------------------

// ─────────────── helper: send a sentence to Piper and play it ─────
// ─────────────── helper: send a sentence to Piper and play it ─────
void say(const std::string &sentence)
{
    json j = {{"text", sentence}};
    std::string jsonEsc = j.dump(); // {"text":"..."} with double-quotes

    // escape every existing double-quote for the shell
    for (std::size_t pos = 0;
         (pos = jsonEsc.find('"', pos)) != std::string::npos;
         pos += 2) // skip over the \" we just added
    {
        jsonEsc.insert(pos++, 1, '\\'); // << fix is here
    }

    std::string cmd =
        "echo \"" + jsonEsc + "\" | ./piper/piper "
                              "--model ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx "
                              "--config ./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json "
                              "--output_file spoken.wav "
                              "--json-input && aplay spoken.wav";

    std::system(cmd.c_str());
}

// ------------------------------------------------------------------

void triggerPythonScript()
{
    std::ofstream triggerFile(TRIGGER_PATH);
    triggerFile << "run\n";
}

bool waitForResponse(std::string &detectedText)
{
    for (int retry = 0; retry < 20; ++retry)
    {
        std::ifstream in(FEEDBACK_PATH);
        if (in.good())
        {
            json j;
            in >> j;
            detectedText = j.value("text", "");
            in.close();
            std::remove(FEEDBACK_PATH.c_str());
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

int main()
{
    std::cout << "[INFO] AI Guidance Assistant started.\n";

    /* >>> NEW: power-on announcement */
    say("Power on, Insight is your assistant");

    while (true)
    {
        triggerPythonScript();
        std::cout << "[INFO] Triggered Python script to capture and process frame.\n";

        std::string detected;
        if (waitForResponse(detected))
        {
            std::cout << "[INFO] Received response: " << detected << '\n';
            say(detected); // speak detection result
        }
        else
        {
            std::cerr << "[ERROR] No response from Python script.\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return 0;
}
