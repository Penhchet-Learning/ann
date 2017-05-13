#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::feedForward() {
  Matrix *a;
  Matrix *b;
  Matrix *c;
  MultiplyMatrix *multiplier;

  for(int i = 0; i < (this->layers.size() - 1); i++) {
    a = this->getNeuronMatrix(i);

    if(i != 0) {
      a = this->getActivatedNeuronMatrix(i);
    }

    b           = this->getWeightMatrix(i);
    multiplier  = new MultiplyMatrix(a, b);
    c           = multiplier->execute();

    for(int c_index = 0; c_index < c->getNumCols(); c_index++) {
      if(i == (this->layers.size() - 2)) {
        this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index) + this->bias);
      } else {
        this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index) + this->bias);
      }
    }

    delete a;
    delete b;
    delete c;
  }
}
