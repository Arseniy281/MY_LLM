#include <vector>
#include <unordered_map>

class Trie {
private:
    struct Node {
        std::unordered_map<char, Node*> children;
        bool is_end_ = false;
    };

    Node root_;

public:
    Trie() = default;
    Trie(const std::vector<std::string>& vocab);

    void Insert(const std::string& word);
    std::string LongestToken(const std::string& text, size_t index);
};