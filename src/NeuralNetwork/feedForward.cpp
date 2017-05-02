#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::feedForward() {
  Matrix *a;
  Matrix *b;
  Matrix *c;

  for(int i = 0; i < (this->layers.size() - 1); i++) {
    a = this->getNeuronMatrix(i);

    if(i != 0) {
      a = this->getActivatedNeuronMatrix(i);
    }

    b = this->getWeightMatrix(i);
    c = (new utils::MultiplyMatrix(a, b))->execute();

    for(int c_index = 0; c_index < c->getNumCols(); c_index++) {
      if(i == (this->layers.size() - 2)) {
        this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index), false);
      } else {
        this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index), false);
      }
    }
  }
}
