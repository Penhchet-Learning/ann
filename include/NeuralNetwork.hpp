#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#define COST_SIMPLE 1
#define COST_RELU 2

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "utils/MultiplyMatrix.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"
#include "../include/json.hpp"

using namespace std;
using namespace utils;
using json = nlohmann::json;

class NeuralNetwork
{
public:
  NeuralNetwork(
    vector<int> topology, 
    string mode,
    double bias = 1,
    double learningRate = 0.05, 
    double momentum = 1
  );

  NeuralNetwork(
    vector<int> topology, 
    string mode,
    int hiddenActivationType,
    int outputActivationType,
    int costFunctionType,
    double bias = 1,
    double learningRate = 0.05, 
    double momentum = 1
  );

  void setCurrentInput(vector<double> input);
  void setCurrentTarget(vector<double> target) { this->target = target; };
  void feedForward();
  void backPropagation();
  void printToConsole();
  void setErrors();

  Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); };
  Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); };
  Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); };
  Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };

  void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); };

  void setTrainingData(string filename);
  void setLabelData(string filename);

  double getTotalError() { return this->error; };
  vector<double> getErrors() { return this->errors; };
  void printInputToConsole();
  void printOutputToConsole();
  void printTargetToConsole();
  void printHistoricalErrors();

  void saveWeights(string filename);  // Saves weights as a json file
  void loadWeights(string filename);  // Load weights from a json file

  void train(
    vector<double> input, 
    vector<double> target, 
    double bias = 1,
    double learningRate = 0.05, 
    double momentum = 1
  );
private:
  int               topologySize;
  int               hiddenActivationType  = 2; // relu for hidden by default
  int               outputActivationType  = 3; // sigmoid for output by default
  int               costFunctionType      = 1; // simple cost function (target - guess)
  vector<int>       topology;
  vector<Layer *>   layers;
  vector<Matrix *>  weightMatrices;
  vector<Matrix *>  gradientMatrices;
  vector<double>    input;
  vector<double>    target;
  double            error;
  double            momentum;
  double            learningRate;
  double            bias;
  vector<double>    errors;
  vector<double>    historicalErrors;

  vector<vector<double> > trainingData;
  vector<vector<double> > labelData;
};

#endif
