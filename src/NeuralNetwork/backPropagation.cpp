#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::backPropagation() {
  // Output to Hidden
  vector<Matrix *> newWeights;
  Matrix *gradient;
  Layer *l;
  Layer *lastHiddenLayer;
  Matrix *weightsOutputToHidden;
  Matrix *deltaOutputToHidden;
  Matrix *newWeightsOutputToHidden;

  // Hidden to input
  Matrix *derivedHidden;
  Matrix *activatedHidden;
  Matrix *derivedGradients;
  Matrix *weightMatrix;
  Matrix *originalWeight;
  Matrix *leftNeurons;
  Matrix *deltaWeights;
  Matrix *newWeightsHidden;


  // output to hidden
  int outputLayerIndex      = this->layers.size() - 1;
  Matrix *derivedValuesYToZ = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
  Matrix *gradientsYToZ     = new Matrix(1, this->layers.at(outputLayerIndex)->getNeurons().size(), false);

  for(int i = 0; i < this->errors.size(); i++) {
    double d  = derivedValuesYToZ->getValue(0, i);
    double e  = this->errors.at(i);
    double g  = d * e;
    gradientsYToZ->setValue(0, i, g);
  }

  int lastHiddenLayerIndex  = outputLayerIndex - 1;
  lastHiddenLayer           = this->layers.at(lastHiddenLayerIndex);
  weightsOutputToHidden     = this->weightMatrices.at(outputLayerIndex - 1);
  deltaOutputToHidden       = (new utils::MultiplyMatrix(
                                gradientsYToZ->transpose(), 
                                lastHiddenLayer->matrixifyActivatedVals()
                              ))->execute()->transpose();

  newWeightsOutputToHidden  = new Matrix(
                                deltaOutputToHidden->getNumRows(), 
                                deltaOutputToHidden->getNumCols(), 
                                false
                              );

  for(int r = 0; r < deltaOutputToHidden->getNumRows(); r++) {
    for(int c = 0; c < deltaOutputToHidden->getNumCols(); c++) {
      double originalWeight = weightsOutputToHidden->getValue(r, c);
      double deltaWeight    = deltaOutputToHidden->getValue(r, c);

      // with momentum
      originalWeight  = this->momentum * originalWeight;

      // with learning rate
      deltaWeight = this->learningRate * deltaWeight;

      newWeightsOutputToHidden->setValue(r, c, (originalWeight - deltaWeight));
    }
  }

  newWeights.push_back(newWeightsOutputToHidden);

  gradient = new Matrix(gradientsYToZ->getNumRows(), gradientsYToZ->getNumCols(), false);
  for(int r = 0; r < gradientsYToZ->getNumRows(); r++) {
    for(int c = 0; c < gradientsYToZ->getNumCols(); c++) {
      gradient->setValue(r, c, gradientsYToZ->getValue(r, c));
    }
  }

  // Moving from last hidden layer down to input layer
  for(int i = (outputLayerIndex - 1); i > 0; i--) {
    l                 = this->layers.at(i);
    derivedHidden     = l->matrixifyDerivedVals();
    activatedHidden   = l->matrixifyActivatedVals();
    derivedGradients  = new Matrix(
                          1,
                          l->getNeurons().size(),
                          false
                        );

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

    leftNeurons = (i - 1) == 0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i - 1)->matrixifyActivatedVals();

    deltaWeights  = (new utils::MultiplyMatrix(derivedGradients->transpose(), leftNeurons))->execute()->transpose();
    newWeightsHidden    = new Matrix(
                            deltaWeights->getNumRows(),
                            deltaWeights->getNumCols(),
                            false
                          );
    
    for(int r = 0; r < newWeightsHidden->getNumRows(); r++) {
      for(int c = 0; c < newWeightsHidden->getNumCols(); c++) {
        double w  = originalWeight->getValue(r, c);
        double d  = deltaWeights->getValue(r, c);

        // with momentum
        w = this->momentum * w;

        // with learninng rate
        d = this->learningRate * d;

        double n  = w - d;
        newWeightsHidden->setValue(r, c, n);
      }
    }

    gradient = new Matrix(derivedGradients->getNumRows(), derivedGradients->getNumCols(), false);
    for(int r = 0; r < derivedGradients->getNumRows(); r++) {
      for(int c = 0; c < derivedGradients->getNumCols(); c++) {
        gradient->setValue(r, c, derivedGradients->getValue(r, c));
      }
    }

    newWeights.push_back(newWeightsHidden);
  }

  reverse(newWeights.begin(), newWeights.end());
  this->weightMatrices = newWeights;
}
