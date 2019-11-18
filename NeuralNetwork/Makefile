.PHONY: clean

sample_learning.out: sample_learning.cpp NeuralNetwork.hpp
	g++ -I /usr/include/eigen3 -o $@ -std=c++11 $^

clean:
	rm *.out
