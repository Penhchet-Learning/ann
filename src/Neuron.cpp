#include "../include/Neuron.hpp"

void Neuron::setVal(double val) {
  this->val = val;
  activate();
  derive();
}

// Constructor
Neuron::Neuron(double val) {
  this->val = val;
  activate();
  derive();
}

// Fast Sigmoid Function
// f(x) = x / (1 + |x|)
void Neuron::activate() {
  this->activatedVal = this->val / (1 + abs(this->val));
}

// Derivative for fast sigmoid function
// f'(x) = f(x) * (1 - f(x))
void Neuron::derive() {
  this->derivedVal = this->activatedVal * (1 - this->activatedVal);
}
