#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::train(
  vector<double> input, 
  vector<double> target, 
  double bias,
  double learningRate,
  double momentum 
) {
  this->learningRate  = learningRate;
  this->momentum      = momentum;
  this->bias          = bias;

  clock_t t;
  this->setCurrentInput(input);
  this->setCurrentTarget(target);
  t = clock();
  this->feedForward();
  t = clock() - t;
  //printf ("FF: %f seconds.\n",t,((float)t)/CLOCKS_PER_SEC);
  this->setErrors();

  t = clock();
  this->backPropagation();
  t = clock() - t;
  //printf ("BP: %f seconds.\n",t,((float)t)/CLOCKS_PER_SEC);
}
