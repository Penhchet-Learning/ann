#include <iostream>
#include <vector>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"

using namespace std;

int main(int argc, char **argv) {
  vector<int> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(3);

  vector<double> input;
  input.push_back(1.0);
  input.push_back(0.0);
  input.push_back(1.0);

  NeuralNetwork *nn = new NeuralNetwork(topology);
  nn->setCurrentInput(input);

  Matrix *a = nn->getNeuronMatrix(0);
  cout << "A" << endl;
  a->printToConsole();
  cout << endl;
  cout << "B" << endl;
  Matrix *b = nn->getWeightMatrix(0);
  b->printToConsole();
  cout << endl;

  Matrix *c = (new utils::MultiplyMatrix(a, b))->execute();

  cout << "C" << endl;
  c->printToConsole();
  cout << endl;

  Matrix *d = nn->getWeightMatrix(1);
  Matrix *e = (new utils::MultiplyMatrix(c, d))->execute();

  cout << "D" << endl;
  d->printToConsole();
  cout << endl;

  cout << "E" << endl;
  e->printToConsole();
  cout << endl;

  return 0;
}
