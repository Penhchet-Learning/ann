#include "../include/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
  if(this->target.size() == 0) {
    cerr << "No target for this neural network" << endl;
    assert(false);
  }

  if(this->target.size() != this->layers.at(this->layers.size() - 1)->getNeurons().size()) {
    cerr << "Target size is not the same as output layer size: " << this->layers.at(this->layers.size() - 1)->getNeurons().size() << endl;
    assert(false);
  }

  this->error = 0.00;
  int outputLayerIndex  = this->layers.size() - 1;
  vector<Neuron *> outputNeurons  = this->layers.at(outputLayerIndex)->getNeurons();
  for(int i = 0; i < target.size(); i++) {
    double tempErr  = (outputNeurons.at(i)->getActivatedVal() - target.at(i));
    errors.at(i) = tempErr;
    this->error += tempErr;
  }

  historicalErrors.push_back(this->error);
}

void NeuralNetwork::feedForward() {
  for(int i = 0; i < (this->layers.size() - 1); i++) {
    Matrix *a = this->getNeuronMatrix(i);

    if(i != 0) {
      a = this->getActivatedNeuronMatrix(i);
    }

    Matrix *b = this->getWeightMatrix(i);
    Matrix *c = (new utils::MultiplyMatrix(a, b))->execute();

    for(int c_index = 0; c_index < c->getNumCols(); c_index++) {
      this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index));
    }
  }
}

void NeuralNetwork::printToConsole() {
  for(int i = 0; i < this->layers.size(); i++) {
    cout << "LAYER: " << i << endl;
    if(i == 0) {
      Matrix *m = this->layers.at(i)->matrixifyVals();
      m->printToConsole();
    } else {
      Matrix *m = this->layers.at(i)->matrixifyActivatedVals();
      m->printToConsole();
    }
    cout << "======================" << endl;
    if(i < this->layers.size() - 1) {
      cout << "Weight Matrix: " << i << endl;
      this->getWeightMatrix(i)->printToConsole();
    }
    cout << "======================" << endl;
  }
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
  this->input = input;

  for(int i = 0; i < input.size(); i++) {
    this->layers.at(0)->setVal(i, input.at(i));
  }
}

NeuralNetwork::NeuralNetwork(vector<int> topology) {
  this->topology      = topology;
  this->topologySize  = topology.size();

  for(int i = 0; i < topologySize; i++) {
    Layer *l  = new Layer(topology.at(i));
    this->layers.push_back(l);
  }

  for(int i = 0; i < (topologySize - 1); i++) {
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
  }

  this->error = 0.00;
}
