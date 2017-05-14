#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::loadWeights(string filename) {
  std::ifstream i(filename);
  json j;
  i >> j;

  vector< vector< vector<double> > > temp = j["weights"];
}
