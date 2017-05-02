#include <iostream>
#include <vector>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
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
  string aveHistErrorFile = configJson["aveHistErrorFile"];
  double momentum        = configJson["momentum"];
  double learningRate    = configJson["learningRate"];

  cout << "Initializing neural network..." << endl;
  NeuralNetwork *nn = new NeuralNetwork(topology, momentum, learningRate);
  cout << "Done initializing neural network..." << endl;

  cout << "Starting training..." << endl;
  int epoch = 1;

  vector<double> histAveError;
  while(epoch <= epochThreshold) {
    double aveError = 0;

    vector<vector<double> > data  = (new utils::FetchCSVData(trainingData))->execute();
    for(int i = 0; i < data.size(); i++) {
      nn->train(data.at(i), data.at(i));
      //cout << "Error for DP " << i << ": " << nn->getTotalError() << "\r";
      aveError += nn->getTotalError();
    }

    aveError = aveError / data.size();

    histAveError.push_back(aveError);
    cout << aveError << endl;

    epoch++;
  }

  ofstream writer;
  writer.open(aveHistErrorFile);
  for(int i = 0; i < histAveError.size(); i++) {
    writer << histAveError.at(i);
    if(i != histAveError.size() - 1) {
      writer << ",";
    }
  }
  writer << endl;
  writer.close();

  return 0;
}
