template <class Scalar, class Index>
Network<Scalar, Index>::Network(
    Index numRotor,
    const std::vector<std::tuple<Index, Index, Scalar>> &edgeList)
    : numRotor_(numRotor), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::domain_error(
            "==Cnqs::Problem::Problem== Need at least two quantum rotors");
    }

    for (auto &edge : edgeList_) {
        Index j = std::get<0>(edge);
        Index k = std::get<1>(edge);
        Scalar g = std::get<2>(edge);

        if (j == k) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Self-loops are not allowed in "
                "network");
        }

        // switch order to ensure j < k
        if (j > k) {
            Index temp = j;
            j = k;
            k = temp;

            edge = std::make_tuple(j, k, g);
        }

        if (j < 0 || k >= numRotor_) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Edge specification is not valid");
        }
    }
}

template <class Scalar, class Index>
Network<Scalar, Index>::Network(const std::string &networkFileName) {
    nlohmann::json description;

    {
        std::ifstream networkFile(networkFileName);
        networkFile >> description;
    }

    const auto &edgeList = description["edges"];

    numRotor_ = description["num_rotor"];
    edgeList_.reserve(edgeList.size());

    for (const auto &edge : edgeList) {
        Index j = edge["node1"];
        Index k = edge["node2"];
        const Scalar g = edge["weight"];

        // switch order to ensure j < k
        if (j > k) {
            const Index temp = j;
            j = k;
            k = temp;
        }

        if (j < 0 || k >= numRotor_) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Edge specification is not valid");
        }

        edgeList_.emplace_back(j, k, g);
    }
}

template <class Scalar, class Index>
Scalar Network<Scalar, Index>::eigValLowerBound() const {
    Scalar mu = -1.0e-09;

    for (const auto &edge : edgeList_) {
        mu -= std::abs(std::get<2>(edge));
    }

    return mu;
}

template <class Scalar, class Index>
nlohmann::json Network<Scalar, Index>::description() const {
    nlohmann::json description;

    description["num_rotor"] = numRotor_;
    description["edges"] = nlohmann::json::array();

    for (const auto &edge : edgeList_) {
        nlohmann::json edgeJson;

        edgeJson["node1"] = std::get<0>(edge);
        edgeJson["node2"] = std::get<1>(edge);
        edgeJson["weight"] = std::get<2>(edge);

        description["edges"].push_back(edgeJson);
    }

    return description;
}
