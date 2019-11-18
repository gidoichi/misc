#include <stdio.h>
#include <array>
#include <cmath>
#include <new>
#include <random>
#include <iostream>
#include <Eigen/Dense>

class NeuralNetwork
{
private:
  // 入力層のユニット数
  const std::size_t input_units_length_ = 3;
  // 出力層のユニット数
  // const std::size_t output_units_length_ = 1;
  // 中間層のユニット数
  std::size_t middle_units_length_;
  // weights_[k](j, i): 中間層(k-1)のi番目のニューロンと
  // 中間層kのj番目のニューロンをつなぐ重み。
  // ただし、-1番目の層を入力層、1番目の層を出力層とする。
  std::array<Eigen::MatrixXd, 2> weights_;
  // 乱数生成用
  std::random_device rnd;
  // 活性化関数
  double activation_function(double x);
  // 1行目だけ定数関数となっている活性化関数
  Eigen::MatrixXd activation_function(Eigen::MatrixXd x);
  // 活性化関数の導関数
  double derivative_activation_function(double x);
  // 1行目だけ定数関数となっている活性化関数の導関数
  Eigen::MatrixXd derivative_activation_function(Eigen::MatrixXd x);


public:
  // 学習率
  double learning_rate = 0.01;

  NeuralNetwork(int middle_units_length);
  // 学習
  void fit(double obj_value, std::array<double, 2> exp_values);
  // 予想
  double predict(std::array<double, 2> exp_values);
  // 重みの表示
  void show_weights();
};

NeuralNetwork::NeuralNetwork(int middle_units_length)
  : middle_units_length_(middle_units_length+1)
{
  // 必要な領域の確保
  new(&weights_[0]) Eigen::MatrixXd(middle_units_length_, input_units_length_);
  new(&weights_[1]) Eigen::MatrixXd(1, middle_units_length_);

  // 重みの初期化
  std::mt19937 mt(rnd());
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < weights_[k].rows(); ++j) {
      for (int i = 0; i < weights_[k].cols(); ++i) {
        double max_abs = std::sqrt(weights_[k].cols());
        weights_[k](j, i) = (double)mt() / (((long long)1<<32)-1) * 2.0/max_abs
          - 1.0/max_abs;
      }
    }
  }
}

double NeuralNetwork::activation_function(double x)
{
  // ReLU
  return (x<=0) ? 0 : x;
  // シグモイド関数
  // double beta = 1;
  // return 1.0 / (1.0 + std::exp(-beta*x));
}

Eigen::MatrixXd NeuralNetwork::activation_function(Eigen::MatrixXd x)
{
  for (int j = 0; j < x.rows(); ++j) {
    for (int i = 0; i < x.cols(); ++i) {
      if (j == 0) {
        x(j, i) = 1;
      } else {
        x(j, i) = activation_function(x(j, i));
      }
    }
  }
  return x;
}

double NeuralNetwork::derivative_activation_function(double x)
{
  return (x<0) ? 0 : 1;
  // double beta = 1;
  // return beta*std::exp(-beta*x) / std::pow((1.0 + std::exp(-beta*x)), 2.0);
}

Eigen::MatrixXd NeuralNetwork::derivative_activation_function(Eigen::MatrixXd x)
{
  for (int j = 0; j < x.rows(); ++j) {
    for (int i = 0; i < x.cols(); ++i) {
      if (j == 0) {
        x(j, i) = 0;
      } else {
        x(j, i) = derivative_activation_function(x(j, i));
      }
    }
  }
  return x;
}

void NeuralNetwork::fit(double obj_value, std::array<double, 2> exp_values)
{
  std::array<Eigen::MatrixXd, 3> inputs = {
                                           Eigen::MatrixXd(input_units_length_, 1),
                                           Eigen::MatrixXd(middle_units_length_, 1),
                                           Eigen::MatrixXd(1, 1),
  };

  // 各ニューロンへの入力和を計算
  for (int i = 0; i <= exp_values.size(); ++i) {
    inputs[0](i, 0) = (i<exp_values.size()) ? exp_values[i] : 1;
  }
  inputs[1] = weights_[0] * inputs[0];
  inputs[2] = weights_[1] * activation_function(inputs[1]);

  // 重みの更新
  Eigen::MatrixXd teacher(1, 1);
  teacher(0, 0) = obj_value;
  std::array<Eigen::MatrixXd, 2> delta;
  delta[1] = (teacher - inputs[2]);
  delta[0] = (weights_[1].transpose() * delta[1]).array()
    * derivative_activation_function(inputs[1]).array();
  weights_[1] += learning_rate * delta[1] * activation_function(inputs[1]).transpose();
  weights_[0] += learning_rate * delta[0] * inputs[0].transpose();
}

double NeuralNetwork::predict(std::array<double, 2> exp_values)
{
  Eigen::MatrixXd input(exp_values.size() + 1, 1);
  for (int i = 0; i <= exp_values.size(); ++i) {
    input(i, 0) = (i<exp_values.size()) ? exp_values[i] : 1;
  }
  return (weights_[1] * activation_function(weights_[0] * input))(0, 0);
}

void NeuralNetwork::show_weights()
{
  for (int k = 0; k < 2; ++k) {
    printf("weights between layer %d - %d:\n", k-1, k);
    std::cout << weights_[k] << std::endl;
  }
}
