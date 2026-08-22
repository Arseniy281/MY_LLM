#include "bpe_tokenizer.h"
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

std::vector<std::vector<std::string>> BPETokenizer::SplitInVector(const std::string& str) {
    std::vector<std::vector<std::string>> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        std::vector<std::string> word;
        for (const auto& letter : token) {
            word.push_back(std::string(1, letter));
        }
        tokens.push_back(word);
    }
    return tokens;
}

std::vector<std::string> BPETokenizer::Split(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

BPETokenizer::BPETokenizer() {
    for (const auto& symbol :
         "abcdefghijklmnopqrstuvwxyz"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "0123456789"
         "!?.,@#%&*-+= ") {

        AddToken(std::string(1, symbol));
    }

    AddToken("\n");
    AddToken("<UNK>");
}

void BPETokenizer::AddToken(const std::string& token) {
    token_to_id_[token] = vocab_.size();
    vocab_.push_back(token);
    trie.Insert(token);
}

std::string BPETokenizer::GetBestPair(const std::unordered_map<std::string, int>& pair_counts) {
    std::string best_pair = "";
    int best_freq = 0;
    for (const auto& [pair, freq] : pair_counts) {
        if (freq > best_freq) {
            best_freq = freq;
            best_pair = pair;
        }
    }
    return best_pair;
}

void BPETokenizer::UpdateWords(std::vector<std::vector<std::string>>& words, const std::string& best_pair) {
    for (auto& word : words) {
        size_t i = 0;
        while (i + 1 < word.size()) {
            if (word[i] + word[i + 1] == best_pair) {
                word[i] = best_pair;
                word.erase(word.begin() + i + 1);
            } else {
                i++;
            }
        }
    }
}

void BPETokenizer::Train(const std::string& corpus, size_t vocab_size) {
    std::vector<std::vector<std::string>> words = SplitInVector(corpus);
    while (vocab_.size() < vocab_size) {
        std::unordered_map<std::string, int> pair_counts;
        for (const auto& word : words) {
            for (size_t i = 0; i < word.size() - 1; i++) {
                std::string pair = word[i] + word[i + 1];
                pair_counts[pair]++;
            }
        }

        std::string best_pair = GetBestPair(pair_counts);
        if (best_pair.size() == 0) {
            return;
        }

        AddToken(best_pair);
        UpdateWords(words, best_pair);
    }
}

void BPETokenizer::Save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    for (const auto& token : vocab_) {
        file << token << "\n";
    }
    file.close();
};

void BPETokenizer::Load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    vocab_.clear();
    token_to_id_.clear();
    std::string token;
    while (std::getline(file, token)) {
        if (!token.empty()) {
            AddToken(token);
        }
    }
    file.close();
}


std::vector<size_t> BPETokenizer::Encode(const std::string& str) {
    std::vector<size_t> ids;

    size_t i = 0;

    while (i < str.size()) {
        std::string prefix = trie.LongestToken(str, i);

        if (!prefix.empty()) {
            ids.push_back(token_to_id_[prefix]);
            i += prefix.size();
        } else {
            ids.push_back(token_to_id_["<UNK>"]);
            i++;
        }
    }

    return ids;
}

std::string BPETokenizer::Decode(std::vector<size_t> ids) {
    std::string result = "";
    for (const auto& id : ids) {
        result += vocab_[id];
    }
    return result;
}

size_t BPETokenizer::GetVocabSize() const {
    return vocab_.size();
}

size_t BPETokenizer::GetTokenId(const std::string& token) const {
    return token_to_id_.at(token);
}