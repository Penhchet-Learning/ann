#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "utils/MultiplyMatrix.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;

class NeuralNetwork
{
public:
  NeuralNetwork(vector<int> topology, double momentum, double learningRate);
  void setCurrentInput(vector<double> input);
  void setCurrentTarget(vector<double> target) { this->target = target; };
  void feedForward();
  void backPropagation();
  void printToConsole();
  void setErrors();

  Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); };
  Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); };
  Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); };
  Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); };

  void setNeuronValue(int indexLayer, int indexNeuron, double val, bool isOutputLayer) { this->layers.at(indexLayer)->setVal(indexNeuron, val, isOutputLayer); };

  void setTrainingData(string filename);
  void setLabelData(string filename);

  double getTotalError() { return this->error; };
  vector<double> getErrors() { return this->errors; };
  void printInputToConsole();
  void printOutputToConsole();
  void printTargetToConsole();
  void printHistoricalErrors();
  void train(vector<double> input, vector<double> target);
private:
  int               topologySize;
  vector<int>       topology;
  vector<Layer *>   layers;
  vector<Matrix *>  weightMatrices;
  vector<Matrix *>  gradientMatrices;
  vector<double>    input;
  vector<double>    target;
  double             error;
  double             momentum;
  double             learningRate;
  vector<double>    errors;
  vector<double>    historicalErrors;

  vector<vector<double> > trainingData;
  vector<vector<double> > labelData;
};

#endif
