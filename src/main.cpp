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
#include "../include/utils/MultiplyMatrix.hpp"
#include "../include/utils/FetchCSVData.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv) {
  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  auto configJson = json::parse(str);

  vector<int> topology  = configJson["topology"];
  int epochThreshold    = configJson["epochThreshold"];
  string trainingData   = configJson["trainingData"];
  string labelData      = configJson["labelData"];
  double momentum       = configJson["momentum"];
  double learningRate   = configJson["learningRate"];
  double bias           = configJson["bias"];

  cout << "Initializing neural network..." << endl;
  NeuralNetwork *nn = new NeuralNetwork(topology);
  cout << "Done initializing neural network..." << endl;

  cout << "Starting training..." << endl;
  int epoch = 1;

  vector<double> histAveError;
  clock_t t;
  while(epoch <= epochThreshold) {
    double aveError = 0;

    vector<vector<double> > data  = (new utils::FetchCSVData(trainingData))->execute();
    vector<vector<double> > labels = (new utils::FetchCSVData(labelData))->execute();
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

    epoch++;
  }

  return 0;
}
