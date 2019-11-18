#include <stdio.h>
#include <iostream>
#include <cmath>
#include <random>
#include "NeuralNetwork.hpp"

int main(void) {
  std::mt19937 mt((unsigned)time(NULL));
  NeuralNetwork nn(1024);
  nn.learning_rate = 0.1;

  // 学習
  int num_of_learning = 100000;
  for (int i = 0; i < num_of_learning; ++i) {
    std::array<double, 2> x;
    if ((i+1) % (num_of_learning / 10) == 0) {
      std::cerr << (i+1) / (num_of_learning / 100) << "%" << std::endl;
    }
    x[0] = (double)mt() / (double)(((long long)1<<32)-1);
    x[1] = (double)mt() / (double)(((long long)1<<32)-1);
    double z = (1.0+std::sin(4.0*M_PI*x[0])) * (x[1]-0.5) / 2.0 + 0.5;
    nn.fit(z, x);
  }

  // 予測
  std::array<double, 2> x;
  for (int x0 = 0; x0 < 50; ++x0) {
    for (int x1 = 0; x1 < 50; ++x1) {
      std::array<double, 2> x = {(double)x0/50.0, (double)x1/50.0};
      printf("%f %f %f\n", nn.predict(x), x[0], x[1]);
    }
  }

  return 0;
}
