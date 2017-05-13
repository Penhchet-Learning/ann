#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
#include <math.h>
using namespace std;

class Neuron
{
public:

  Neuron(double val);

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
};

#endif
