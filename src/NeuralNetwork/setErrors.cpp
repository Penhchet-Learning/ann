#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
  if(this->target.size() == 0) {
    cerr << "No target for this neural network" << endl;
    assert(false);
  }

  if(this->target.size() != this->layers.at(this->layers.size() - 1)->getNeurons().size()) {
    cerr << "Target size (" << this->target.size() << ") is not the same as output layer size: " << this->layers.at(this->layers.size() - 1)->getNeurons().size() << endl;
    for(int i = 0; i < this->target.size(); i++) {
      cout << this->target.at(i) << endl;
    }
    assert(false);
  }

  this->error = 0.00;
  int outputLayerIndex  = this->layers.size() - 1;
  vector<Neuron *> outputNeurons  = this->layers.at(outputLayerIndex)->getNeurons();
  for(int i = 0; i < target.size(); i++) {
    double tempErr  = (outputNeurons.at(i)->getActivatedVal() - target.at(i));
    //double tempErr  = (target.at(i) - outputNeurons.at(i)->getActivatedVal());
    errors.at(i) = tempErr;
    this->error += pow(tempErr, 2);
  }

  this->error = 0.5 * this->error;

  historicalErrors.push_back(this->error);
}

