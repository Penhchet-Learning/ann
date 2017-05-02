#ifndef _MULTIPLY_MATRIX_HPP_
#define _MULTIPLY_MATRIX_HPP_

#include <iostream>
#include <vector>
#include <assert.h>
#include <iomanip> 

#include "../Matrix.hpp"

using namespace std;

namespace utils 
{
  class MultiplyMatrix
  {
  public:
    MultiplyMatrix(Matrix *a, Matrix *b);

    Matrix *execute();
  private:
    Matrix *a;
    Matrix *b;
    Matrix *c;
  };
}

#endif
