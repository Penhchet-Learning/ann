#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;

class NeuralNetwork
{
public:
  NeuralNetwork(vector<int> topology);
  void setCurrentInput(vector<double> input);
  void printToConsole();
  void feedForward();

  Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); };
  Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); };
  Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); };
  Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); };
private:
  int               topologySize;
  vector<int>       topology;
  vector<Layer *>   layers;
  vector<Matrix *>  weightMatrices;
  vector<double>    input;
};

#endif
