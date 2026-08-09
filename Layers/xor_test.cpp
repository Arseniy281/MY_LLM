#include <iostream>
#include "../Tensor/tensor.h"
#include "linear_layer.h"
#include "mse_loss.h"
#include "../Autograd/relu.h"
#include "../Autograd/sigmoid.h"
#include "../Autograd/tanh_op.h"

class XORModel {
public:
    LinearLayer layer1;
    TanhOp tanh1;
    LinearLayer layer2;
    TanhOp tanh2;
    MSELoss mse_loss;

    std::vector<std::shared_ptr<Tensor>> z1_list, a1_list, z2_list, y_pred_list;

    XORModel() : layer1(2, 4), layer2(4, 1) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto z1 = layer1.forward(x);
        auto a1 = tanh1.forward({z1});
        auto z2 = layer2.forward(a1);
        auto y_pred = tanh2.forward({z2});
        
        z1_list.push_back(z1);
        a1_list.push_back(a1);
        z2_list.push_back(z2);
        y_pred_list.push_back(y_pred);
        
        return y_pred;
    }

    void ClearGrad() {
        layer1.ClearGrad();
        layer2.ClearGrad();
        z1_list.clear();
        a1_list.clear();
        z2_list.clear();
        y_pred_list.clear();
    }

    void Update(float lr) {
        layer1.Update(lr);
        layer2.Update(lr);
    }
};

int main() {
    std::cout << "=== XOR Test Started ===\n";
    
    // Данные для XOR (повтор для стабильности)
    std::vector<std::shared_ptr<Tensor>> x_data = {
        std::make_shared<Tensor>(Tensor({1, 2}, {0.0f, 0.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {0.0f, 0.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {0.0f, 1.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {0.0f, 1.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {1.0f, 0.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {1.0f, 0.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {1.0f, 1.0f})),
        std::make_shared<Tensor>(Tensor({1, 2}, {1.0f, 1.0f}))
    };
    
    // Правильные ответы (0 или 1)
    std::vector<std::shared_ptr<Tensor>> y_true = {
        std::make_shared<Tensor>(Tensor({1, 1}, 0.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 0.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 1.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 1.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 1.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 1.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 0.0f)),
        std::make_shared<Tensor>(Tensor({1, 1}, 0.0f))
    };
    
    XORModel model;
    float learning_rate = 0.1f;
    int epochs = 1000;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < x_data.size(); ++i) {
            auto y_pred = model.forward(x_data[i]);
            auto loss = model.mse_loss.forward(y_pred, y_true[i]);
            total_loss += loss->at(0);
            
            Tensor grad_output({1, 1}, 1.0f);
            loss->backward(grad_output);
        }
        
        model.Update(learning_rate);
        model.ClearGrad();

        for (auto& x : x_data) {
            x->ClearGrad();
        }

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / x_data.size() << "\n";
        }
    }
    
    std::cout << "=== XOR Test Finished ===\n";

    std::cout << "\n=== Predictions after training ===\n";
    for (size_t i = 0; i < x_data.size(); ++i) {
        auto y_pred = model.forward(x_data[i]);
        std::cout << "Input: ";
        x_data[i]->print();
        std::cout << "Prediction: ";
        y_pred->print();
        std::cout << "Target: ";
        y_true[i]->print();
        std::cout << "\n";
    }

    return 0;
}