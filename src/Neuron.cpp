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

Neuron::Neuron(double val, int activationType) {
  this->val = val;
  this->activationType = activationType;
  activate();
  derive();
}

// Fast Sigmoid Function
// f(x) = x / (1 + |x|)
void Neuron::activate() {
  //this->activatedVal = this->val / (1 + abs(this->val));
  if(activationType == TANH) {
    this->activatedVal = tanh(this->val);
  } else if(activationType == RELU) {
    if(this->val > 0) {
      this->activatedVal = this->val;
    } else {
      this->activatedVal = 0;
    }
  } else {
    this->activatedVal = tanh(this->val);
  }
}

// Derivative for fast sigmoid function
// f'(x) = f(x) * (1 - f(x))
void Neuron::derive() {
  //this->derivedVal = this->activatedVal * (1 - this->activatedVal);
  if(activationType == TANH) {
    this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
  } else if(activationType == RELU) {
    if(this->val > 0) {
      this->activatedVal = 1;
    } else {
      this->activatedVal = 0;
    }
  } else {
    this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
  }
}
