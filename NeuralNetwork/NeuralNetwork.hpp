#include <stdio.h>
#include <array>
#include <cmath>
#include <new>
#include <random>
#include <vector>
#include <iostream>

template <std::size_t N>
class NeuralNetwork
{
private:
  // 入力層のユニット数
  const std::size_t input_units_length_ = 3;
  // 出力層のユニット数
  // const std::size_t output_units_length_ = 1;
  // 中間層のユニット数
  const std::size_t middle_units_length_ = N;
  // weights_[k][j][i]: 中間層(k-1)のi番目のニューロンと
  // 中間層kのj番目のニューロンをつなぐ重み。
  // ただし、-1番目の層を入力層、1番目の層を出力層とする。
  std::vector< std::vector< std::vector<double> > > weights_;
  // 学習率
  double learning_rate_ = 0.01;
  // 活性化関数
  double activation_function(double x);
  // 活性化関数の導関数
  double derivative_activation_function(double x);

public:
  // 慣性項の寄与率
  double inertia_weight = 0;

  NeuralNetwork();
  // 学習
  void fit(double obj_value, std::array<double, 2> exp_values);
  // 予想
  double predict(std::array<double, 2> exp_values);
  // 重みの表示
  void show_weights();
};

template<std::size_t N>
NeuralNetwork<N>::NeuralNetwork()
{
  // 必要な領域の確保
  weights_.resize(2);
  weights_[0].resize(middle_units_length_);
  for (int i = 0; i < middle_units_length_; ++i) {
    weights_[0][i].resize(input_units_length_);
  }
  weights_[1].resize(1);
  weights_[1][0].resize(middle_units_length_);

  // 重みの初期化
  std::mt19937 mt((unsigned)time(NULL));
  for (int k = 0; k < weights_.size(); ++k) {
    for (int j = 0; j < weights_[k].size(); ++j) {
      for (int i = 0; i < weights_[k][j].size(); ++i) {
        weights_[k][j][i] = (double)mt() / (((long long)1<<32)-1) * 2/weights_[k][j].size()
          - 1.0/weights_[k][j].size();
      }
    }
  }
}

template<std::size_t N>
double NeuralNetwork<N>::activation_function(double x)
{
  // ReLU
  return (x<=0) ? 0 : x;
  // シグモイド関数
  // return 1.0 / (1.0 + std::exp(-4*x));
}

template<std::size_t N>
double NeuralNetwork<N>::derivative_activation_function(double x)
{
  return (x<0) ? 0 : 1;
  // return 4*std::exp(-4*x) / std::pow((1.0 + std::exp(-4*x)), 2.0);
}

template<std::size_t N>
void NeuralNetwork<N>::fit(double obj_value, std::array<double, 2> exp_values)
{
  static int max_unit_length
    = (input_units_length_ >= middle_units_length_) ?
    input_units_length_ : middle_units_length_;

  // 慣性項
  static double inertia_term[2][N][N] = {{}};

  // 入力層以外のニューロンに入る値の総和
  double neuron[2][max_unit_length] = {{}};

  // 各ニューロンの重みを計算
  for (int k = 0; k < weights_.size(); ++k) {
    for (int j = 0; j < weights_[k].size(); ++j) {
      neuron[k][j] = 0;
      for (int i = 0; i < weights_[k][j].size(); ++i) {
        if (k == 0) {
          neuron[k][j] += weights_[k][j][i] * ((i<exp_values.size()) ? exp_values[i] : 1);
        } else {
          neuron[k][j] += weights_[k][j][i] * activation_function(neuron[k-1][i]);
        }
      }
    }
  }

  // バックプロパゲーション法による重みの更新
  for (int j = 0; j < weights_[0].size(); ++j) {
    for (int i = 0; i < weights_[0][j].size(); ++i) {
      inertia_term[0][j][i]
        = learning_rate_
        * weights_[1][0][i]
        * (obj_value - activation_function(neuron[1][0]))
        * derivative_activation_function(neuron[1][0])
        * derivative_activation_function(neuron[0][i])
        * ((i<exp_values.size()) ? exp_values[i] : 1)
        + inertia_weight * inertia_term[0][j][i];
      weights_[0][j][i] += inertia_term[0][j][i];
    }
  }
  for (int i = 0; i < weights_[1][0].size(); ++i) {
    inertia_term[1][0][i]
      = learning_rate_
      * (obj_value - activation_function(neuron[1][0]))
      * derivative_activation_function(neuron[1][0])
      * neuron[0][i]
      + inertia_weight * inertia_term[1][0][i];
    weights_[1][0][i] += inertia_term[1][0][i];
  }
}

template<std::size_t N>
double NeuralNetwork<N>::predict(std::array<double, 2> exp_values)
{
  static int max_unit_length
    = (input_units_length_ >= middle_units_length_) ?
    input_units_length_ : middle_units_length_;
  double a[2][max_unit_length] = {{}};
  double *pre = a[0], *post = a[1];

  // 出力の計算
  for (int i = 0; i < weights_[0].size(); ++i) {
    pre[i] = (i<exp_values.size()) ? exp_values[i] : 1;
  }
  for (int k = 0; k < weights_.size(); ++k) {
    for (int j = 0; j < weights_[k].size(); ++j) {
      post[j] = 0;
      for (int i = 0; i < weights_[k][j].size(); ++i) {
        post[j] += weights_[k][j][i] * pre[i];
      }
      post[j] = activation_function(post[j]);
    }
    double *t = pre;
    pre = post;
    post = t;
  }

  return pre[0];
}

template<std::size_t N>
void NeuralNetwork<N>::show_weights()
{
  for (int k = 0; k < weights_.size(); ++k) {
    printf("weights between layer %d - %d:\n", k-1, k);
    for (int j = 0; j < weights_[k].size(); ++j) {
      for (int i = 0; i < weights_[k][j].size(); ++i) {
        printf("%s%6.3f", (i==0)?"":" ", weights_[k][j][i]);
      }
      puts("");
    }
    puts("");
  }
}
