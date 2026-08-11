#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include "trie.h"

class BPETokenizer {
private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    Trie trie;

    std::vector<std::vector<std::string>> SplitInVector(const std::string& str);
    std::vector<std::string> Split(const std::string& str);

    void UpdateWords(std::vector<std::vector<std::string>>& words, const std::string& best_pair);
    std::string GetBestPair(const std::unordered_map<std::string, int>& pair_counts);

public:
    BPETokenizer();

    void AddToken(const std::string& token);
    void Train(const std::string& corpus, size_t vocab_size=1000);

    void Save(const std::string& filename);
    void Load(const std::string& filename);

    std::vector<size_t> Encode(const std::string& str);
    std::string Decode(std::vector<size_t> ids);

};