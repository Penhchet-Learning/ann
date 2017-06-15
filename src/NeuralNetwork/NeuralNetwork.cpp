#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::setLabelData(string filename) {
  labelData.clear();
  ifstream infile(filename);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stod(tok));
    }

    labelData.push_back(dRow);
  }
}

void NeuralNetwork::setTrainingData(string filename) {
  trainingData.clear();
  ifstream infile(filename);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stod(tok));
    }

    trainingData.push_back(dRow);
  }
}

void NeuralNetwork::printInputToConsole() {
  for(int i = 0; i < this->input.size(); i++) {
    cout << this->input.at(i) << "\t";
  }

  cout << endl;
}

void NeuralNetwork::printTargetToConsole() {
  for(int i = 0; i < this->target.size(); i++) {
    cout << this->target.at(i) << "\t";
  }

  cout << endl;
}

void NeuralNetwork::printHistoricalErrors() {
  for(int i = 0; i < this->historicalErrors.size(); i++) {
    cout << this->historicalErrors.at(i);
    if(i != this->historicalErrors.size() - 1) {
      cout << ",";
    }
  }
  cout << endl;
}

void NeuralNetwork::printOutputToConsole() {
  int indexOfOutputLayer  = this->layers.size() - 1;
  Matrix *outputValues    = this->layers.at(indexOfOutputLayer)->matrixifyActivatedVals();
  for(int c = 0; c < outputValues->getNumCols(); c++) {
    cout << outputValues->getValue(0, c) << "\t";
  }
  cout << endl;
}

void NeuralNetwork::printToConsole() {
  for(int i = 0; i < this->layers.size(); i++) {
    cout << "LAYER: " << i << endl;
    if(i == 0) {
      Matrix *m = this->layers.at(i)->matrixifyVals();
      m->printToConsole();
    } else {
      Matrix *m = this->layers.at(i)->matrixifyActivatedVals();
      m->printToConsole();
    }
    cout << "======================" << endl;
    if(i < this->layers.size() - 1) {
      cout << "Weight Matrix: " << i << endl;
      this->getWeightMatrix(i)->printToConsole();
    }
    cout << "======================" << endl;
  }
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
  this->input = input;

  for(int i = 0; i < input.size(); i++) {
    this->layers.at(0)->setVal(i, input.at(i));
  }
}

// Constructor 1
NeuralNetwork::NeuralNetwork(
  vector<int> topology, 
  string mode,
  double bias,
  double learningRate, 
  double momentum
) {
  this->topology      = topology;
  this->topologySize  = topology.size();
  this->learningRate  = learningRate;
  this->momentum      = momentum;
  this->bias          = bias;

  // Check for autoencoder mode
  if(mode.compare("autoencoder") == 0) {
    if(this->topology.size() % 2 == 0) {
      cerr << "Invalid topology. Should be odd number in size" << endl;
      exit(-1);
    }
  //} else if(mode.compare("classifier") == 0) {
  } else {
    cerr << "Invalid mode " << mode << endl;
    exit(-1);
  }

  for(int i = 0; i < topologySize; i++) {
    if(i > 0 && i < (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
      this->layers.push_back(l);
    } else if(i == (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->outputActivationType);
      this->layers.push_back(l);
    } else {
      Layer *l  = new Layer(topology.at(i));
      this->layers.push_back(l);
    }
  }

  for(int i = 0; i < (topologySize - 1); i++) {
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
  }

  this->error = 0.00;
}

// Constructor 1
NeuralNetwork::NeuralNetwork(
  vector<int> topology, 
  string mode,
  int hiddenActivationType,
  int outputActivationType,
  int costFunctionType,
  double bias,
  double learningRate, 
  double momentum
) {
  this->topology      = topology;
  this->topologySize  = topology.size();
  this->learningRate  = learningRate;
  this->momentum      = momentum;
  this->bias          = bias;

  this->hiddenActivationType  = hiddenActivationType;
  this->outputActivationType  = outputActivationType;
  this->costFunctionType      = costFunctionType;

  // Check for autoencoder mode
  if(mode.compare("autoencoder") == 0) {
    if(this->topology.size() % 2 == 0) {
      cerr << "Invalid topology. Should be odd number in size" << endl;
      exit(-1);
    }
  //} else if(mode.compare("classifier") == 0) {
  } else {
    cerr << "Invalid mode " << mode << endl;
    exit(-1);
  }

  for(int i = 0; i < topologySize; i++) {
    if(i > 0 && i < (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
      this->layers.push_back(l);
    } else if(i == (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->outputActivationType);
      this->layers.push_back(l);
    } else {
      Layer *l  = new Layer(topology.at(i));
      this->layers.push_back(l);
    }
  }

  for(int i = 0; i < (topologySize - 1); i++) {
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
  }

  this->error = 0.00;
}
