#include <iostream>
#include <vector>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../include/json.hpp"
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/Misc.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv) {
  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  auto configJson = json::parse(str);

  vector<int> topology  = configJson["topology"];
  int epochThreshold    = configJson["epochThreshold"];
  double errorThreshold = configJson["errorThreshold"];
  string trainingData   = configJson["trainingData"];
  string labelData      = configJson["labelData"];
  double momentum       = configJson["momentum"];
  double learningRate   = configJson["learningRate"];
  double bias           = configJson["bias"];
  string mode           = configJson["mode"];
  string outputWeights  = configJson["outputWeights"];
  string weights        = configJson["weights"];
  string validationData = configJson["validationData"];
  string validationLabels = configJson["validationLabels"];

  int hiddenActivationType  = configJson["hiddenActivationType"];
  int outputActivationType  = configJson["outputActivationType"];
  int costFunctionType      = configJson["costFunctionType"];

  cout << "Initializing neural network..." << endl;
  cout << "HIDDEN ACTIVATION: " << hiddenActivationType << endl;
  cout << "OUTPUT ACTIVATION: " << outputActivationType << endl;
  cout << "COST FUNCTION: " << costFunctionType << endl;

  NeuralNetwork *nn = new NeuralNetwork(
                        topology,
                        mode,
                        hiddenActivationType,
                        outputActivationType,
                        costFunctionType
                      );

  // initialize weights if weights file is specified
  if(weights.compare("") != 0) {
    nn->loadWeights(weights);
  }

  cout << "Done initializing neural network..." << endl;

  cout << "Starting training..." << endl;
  int epoch = 1;

  vector<double> histAveError;
  double aveError = 999;
  clock_t t;
  while(epoch <= epochThreshold) {

    vector<vector<double> > data    = utils::Misc::fetchCSVData(trainingData);
    vector<vector<double> > labels  = utils::Misc::fetchCSVData(labelData);
    t = clock();
    for(int i = 0; i < data.size(); i++) {
      nn->train(data.at(i), labels.at(i), bias, learningRate, momentum);
      aveError += nn->getTotalError();
    }

    aveError = aveError / data.size();

    histAveError.push_back(aveError);
    cout << aveError << endl;

    t = clock() - t;
    //printf ("It took %f seconds for a single epoch.\n",t,((float)t)/CLOCKS_PER_SEC);

    // save weights at every iteration
    nn->saveWeights(outputWeights);

    if(aveError < errorThreshold) {
      cout << "Error below threshold" << endl;
      break;
    }

    epoch++;
  }

  return 0;
}
