#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#define TANH 1
#define RELU 2

#include <iostream>
#include <math.h>
using namespace std;

class Neuron
{
public:

  Neuron(double val);
  Neuron(double val, int activationType);

  void setVal(double v);

  void activate();
  void activateFastSigmoid();

  void derive();
  void deriveFastSigmoid();

  // Getter
  double getVal() { return this->val; }
  double getActivatedVal() { return this->activatedVal; }
  double getDerivedVal() { return this->derivedVal; }

private:
  // 1.5
  double val;

  // 0-1
  double activatedVal;

  double derivedVal;

  int activationType = 1;
};

#endif
