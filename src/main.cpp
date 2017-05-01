#include <iostream>
#include <vector>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <streambuf>
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

  cout << "Initializing neural network..." << endl;
  NeuralNetwork *nn = new NeuralNetwork(topology);
  cout << "Done initializing neural network..." << endl;

  cout << "Starting training..." << endl;
  int epoch = 1;
  while(epoch <= epochThreshold) {
    cout << "Epoch: " << epoch << endl;

    vector<vector<double> > data  = (new utils::FetchCSVData(trainingData))->execute();
    cout << "Error:" << endl;
    for(int i = 0; i < data.size(); i++) {
      nn->train(data.at(i), data.at(i));
      cout << nn->getTotalError() << endl;
    }

    cout << "Total Error for epoch " << epoch << ": " << nn->getTotalError() << endl;
    epoch++;
  }

  return 0;
}
