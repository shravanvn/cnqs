#include "Cnqs_Utils.hpp"

std::string Cnqs::padString(const std::string &inputString, int padding) {
    std::string outputString = "";

    for (const auto &c : inputString) {
        outputString.push_back(c);

        if (c == '\n') {
            for (int i = 0; i < padding; ++i) {
                outputString.push_back(' ');
            }
        }
    }

    return outputString;
}
