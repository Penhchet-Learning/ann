#include <iostream>
#include <vector>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"

using namespace std;

int main(int argc, char **argv) {
  vector<double> input;
  input.push_back(1);
  input.push_back(0);
  input.push_back(1);

  vector<int> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(3);

  NeuralNetwork *nn = new NeuralNetwork(topology);
  nn->setCurrentInput(input);
  nn->setCurrentTarget(input);

  // training process
  for(int i = 0; i < 1000; i++) {
  //int i = 0;
  //while(true) {
    cout << "Epoch: " << i << endl;
    nn->feedForward();
    nn->setErrors();
    cout << "Total Error: " << nn->getTotalError() << endl;
    nn->backPropagation();

    cout << "========================" << endl;
    cout << "OUTPUT: ";
    nn->printOutputToConsole();

    cout << "TARGET: ";
    nn->printTargetToConsole();
    cout << "========================" << endl;
    cout << endl;
  }

  nn->printHistoricalErrors();

  return 0;
}
