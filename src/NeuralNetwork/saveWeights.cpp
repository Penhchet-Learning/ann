#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::saveWeights(string filename) {
  json j  = {};

  vector< vector< vector<double> > > weightSet;

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    weightSet.push_back(this->weightMatrices.at(i)->getValues());
  }

  j["weights"]  = weightSet;

  std::ofstream o(filename);
  o << std::setw(4) << j << endl;
}
