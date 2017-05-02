#include "../include/Neuron.hpp"

void Neuron::setVal(double val, bool useFastSigmoid) {
  this->val = val;

  if(useFastSigmoid) {
    activateFastSigmoid();
    deriveFastSigmoid();
  } else {
    activate();
    derive();
  }
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
  //this->activatedVal = this->val / (1 + abs(this->val));
  this->activatedVal = tanh(this->val);
}

void Neuron::activateFastSigmoid() {
  this->activatedVal = this->val / (1 + abs(this->val));
}

// Derivative for fast sigmoid function
// f'(x) = f(x) * (1 - f(x))
void Neuron::derive() {
  //this->derivedVal = this->activatedVal * (1 - this->activatedVal);
  this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
}

// Derivative for fast sigmoid function
// f'(x) = f(x) * (1 - f(x))
void Neuron::deriveFastSigmoid() {
  this->derivedVal = this->activatedVal * (1 - this->activatedVal);
}
