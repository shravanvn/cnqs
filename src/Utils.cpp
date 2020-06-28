#include "Utils.hpp"

std::string PadString(const std::string &input_string, int padding) {
    std::string output_string = "";

    for (const auto &c : input_string) {
        output_string.push_back(c);

        if (c == '\n') {
            for (int i = 0; i < padding; ++i) {
                output_string.push_back(' ');
            }
        }
    }

    return output_string;
}
