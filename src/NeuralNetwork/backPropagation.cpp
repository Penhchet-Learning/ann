#include "../../include/NeuralNetwork.hpp"
#include "../../include/utils/Math.hpp"

void NeuralNetwork::backPropagation() {
  // Output to Hidden
  vector<Matrix *> newWeights;
  Matrix *gradient;
  Matrix *deltaOutputToHidden;
  Matrix *newWeightsOutputToHidden;
  Matrix *derivedValuesYToZ;
  Matrix *gradientsYToZ;

  // Hidden to input
  Matrix *activatedHidden;
  Matrix *derivedGradients;
  Matrix *weightMatrix;
  Matrix *originalWeight;
  Matrix *leftNeurons;
  Matrix *deltaWeights;
  Matrix *newWeightsHidden;

  Matrix *t1;
  Matrix *t2;
  Matrix *t3;
  Matrix *t4;


  // output to hidden
  int outputLayerIndex  = this->layers.size() - 1;
  derivedValuesYToZ     = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
  gradientsYToZ         = new Matrix(
                            1, 
                            this->layers.at(outputLayerIndex)->getNeurons().size(), 
                            false
                          );

  for(int i = 0; i < this->derivedErrors.size(); i++) {
    double d  = derivedValuesYToZ->getValue(0, i);
    double e  = this->derivedErrors.at(i);
    double g  = d * e;
    gradientsYToZ->setValue(0, i, g);
  }

  int lastHiddenLayerIndex  = outputLayerIndex - 1;
  t1  = gradientsYToZ->transpose();
  t2  = this->layers.at(lastHiddenLayerIndex)->matrixifyActivatedVals();
  t3  = new Matrix(
          t1->getNumRows(),
          t2->getNumCols(),
          false
        );

  utils::Math::multiplyMatrix(t1, t2, t3);

  deltaOutputToHidden = t3->transpose();

  delete t1;
  delete t2;
  delete t3;

  int tempR = deltaOutputToHidden->getNumRows();
  int tempC = deltaOutputToHidden->getNumCols();

  newWeightsOutputToHidden  = new Matrix(tempR, tempC, false);

  for(int r = 0; r < tempR; r++) {
    for(int c = 0; c < tempC; c++) {
      double originalWeight = this->weightMatrices.at(outputLayerIndex - 1)->getValue(r, c);
      double deltaWeight    = deltaOutputToHidden->getValue(r, c);

      // with momentum
      originalWeight  = this->momentum * originalWeight;

      // with learning rate
      deltaWeight = this->learningRate * deltaWeight;

      newWeightsOutputToHidden->setValue(r, c, (originalWeight - deltaWeight));
    }
  }

  newWeights.push_back(new Matrix(*newWeightsOutputToHidden));

  int r     = gradientsYToZ->getNumRows();
  int c     = gradientsYToZ->getNumCols();
  gradient  = new Matrix(r, c, false);

  for(int r = 0; r < gradientsYToZ->getNumRows(); r++) {
    for(int c = 0; c < gradientsYToZ->getNumCols(); c++) {
      gradient->setValue(r, c, gradientsYToZ->getValue(r, c));
    }
  }

  // Moving from last hidden layer down to input layer
  for(int i = (outputLayerIndex - 1); i > 0; i--) {
    activatedHidden   = this->layers.at(i)->matrixifyActivatedVals();

    derivedGradients  = new Matrix(
                          1,
                          this->layers.at(i)->getNeurons().size(),
                          false
                        );
    //cout << "derivedGradients: " << derivedGradients->getNumRows() << " x " << derivedGradients->getNumCols() << endl;

    weightMatrix      = this->weightMatrices.at(i);
    originalWeight    = this->weightMatrices.at(i - 1);

    for(int r = 0; r < weightMatrix->getNumRows(); r++) {
      double sum = 0;
      for(int c = 0; c < weightMatrix->getNumCols(); c++) {
        double p = gradient->getValue(0, c) * weightMatrix->getValue(r, c);
        sum += p;
      }

      double g  = sum * activatedHidden->getValue(0, r);
      derivedGradients->setValue(0, r, g);
    }

    //leftNeurons = (i - 1) == 0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i - 1)->matrixifyActivatedVals();
    Matrix *tempLeftNeurons = (i - 1) == 0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i - 1)->matrixifyActivatedVals();
    leftNeurons = tempLeftNeurons->transpose();
    delete tempLeftNeurons;

    t1          = derivedGradients->copy();
    t2          = new Matrix(
                    leftNeurons->getNumRows(),
                    t1->getNumCols(),
                    false
                  );


    //cout << "leftNeurons: " << leftNeurons->getNumRows() << " x " << leftNeurons->getNumCols() << endl;
    //cout << "t1: " << t1->getNumRows() << " x " << t1->getNumCols() << endl;

    utils::Math::multiplyMatrix(leftNeurons, t1, t2);

    //cout << "t2: " << t2->getNumRows() << " x " << t2->getNumCols() << endl;

    deltaWeights  = t2->copy();

    delete t1;
    delete t2;

    int tempR = deltaWeights->getNumRows();
    int tempC = deltaWeights->getNumCols();

    newWeightsHidden    = new Matrix(
                            tempR,
                            tempC,
                            false
                          );

    /*
    cout << "deltaWeights: " << tempR << " x " << tempC << endl;
    cout << endl;
    cout << "originalWeight: " << originalWeight->getNumRows() << " x " << originalWeight->getNumCols() << endl;
    cout << endl;
    */

    for(int r = 0; r < tempR; r++) {
      for(int c = 0; c < tempC; c++) {
        double w  = originalWeight->getValue(r, c);
        double d  = deltaWeights->getValue(r, c);

        // with momentum
        //w = this->momentum * w;

        // with learninng rate
        //d = this->learningRate * d;

        double n  = w - d;
        newWeightsHidden->setValue(r, c, n);
      }
    }


    tempR = derivedGradients->getNumRows();
    tempC = derivedGradients->getNumCols();

    //cout << "X: " << tempR << " x " << tempC << endl;

    delete gradient;
    gradient = new Matrix(tempR, tempC, false);

    for(int r = 0; r < tempR; r++) {
      for(int c = 0; c < tempC; c++) {
        gradient->setValue(r, c, derivedGradients->getValue(r, c));
      }
    }

    newWeights.push_back(new Matrix(*newWeightsHidden));

    delete derivedGradients;
    delete deltaWeights;
    delete leftNeurons;
  }
 
  // deallocate weightMatrices
  for(int i = 0; i < this->weightMatrices.size(); i++) {
    delete this->weightMatrices[i];
  }

  this->weightMatrices.clear();

  reverse(newWeights.begin(), newWeights.end());
  
  // create copies of newWeights
  for(int i = 0; i < newWeights.size(); i++) {
    this->weightMatrices.push_back(new Matrix(*newWeights[i]));
    delete newWeights[i];
  }

  newWeights.clear();

  // cleanup
  delete derivedValuesYToZ;
  delete gradientsYToZ;
  delete gradient;
  delete deltaOutputToHidden;
  delete newWeightsOutputToHidden;
  delete newWeightsHidden;
}
