#include "trie.h"
#include <string>
#include <vector>

Trie::Trie(const std::vector<std::string>& vocab) {
    for (const auto& word : vocab) {
        Insert(word);
    }
}

void Trie::Insert(const std::string& word) {
    Node* cur_node = &root_;
    for (const auto& letter : word) {
        if (cur_node->children.find(letter) == cur_node->children.end()) {
            cur_node->children[letter] = new Node();
        }
        cur_node = cur_node->children[letter];
    }
    cur_node->is_end_ = true;
}

std::string Trie::LongestToken(const std::string& text, size_t index) {
    Node* cur_node = &root_;
    std::string ans = "";
    std::string cur_prefix = "";

    for (size_t i = index;i < text.size(); i++) {
        if (cur_node->children.find(text[i]) == cur_node->children.end()) {
            return ans;
        } else {
            cur_node = cur_node->children[text[i]];
            cur_prefix += text[i];
            if (cur_node->is_end_) {
                ans = cur_prefix;
            }
        }
    } 
    return ans;
}