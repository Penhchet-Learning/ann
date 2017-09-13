#include <iostream>
#include <vector>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../include/json.hpp"
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/Misc.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv) {
  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  auto configJson = json::parse(str);

  vector<int> topology      = configJson["topology"];
  string mode               = configJson["mode"];
  string weights            = configJson["weights"];
  string dataFile           = configJson["dataFile"];
  string outputData         = configJson["outputData"];

  int hiddenActivationType  = configJson["hiddenActivationType"];
  int outputActivationType  = configJson["outputActivationType"];
  int costFunctionType      = configJson["costFunctionType"];

  double bias               = configJson["bias"];

  cout << "Initializing neural network..." << endl;
  cout << "HIDDEN ACTIVATION: " << hiddenActivationType << endl;
  cout << "OUTPUT ACTIVATION: " << outputActivationType << endl;
  cout << "COST FUNCTION: " << costFunctionType << endl;

  NeuralNetwork *nn = new NeuralNetwork(
                        topology,
                        mode,
                        hiddenActivationType,
                        outputActivationType,
                        costFunctionType
                      );

  // initialize weights if weights file is specified
  nn->loadWeights(weights);

  cout << "Done initializing neural network..." << endl;

  cout << "Starting feed forward..." << endl;

  vector<vector<double> > data    = utils::Misc::fetchCSVData(dataFile);

  cout << "Data size: " << data.size() << endl;

  int featureIndex  = topology.size() / 2;

  ofstream fs;
  fs.open(outputData.c_str());

  for(int i = 0; i < data.size(); i++) {
    cout << "Extracting datapoint " << i << endl;
    vector<double> input  = data.at(i);

    nn->bias          = 0;
    nn->setCurrentInput(input);
    nn->feedForward();

    vector<double> dataLine = nn->getActivatedVals(featureIndex);
    for(int j = 0; j < dataLine.size(); j++) {
      //cout << dataLine.at(j) << endl;
      fs << dataLine.at(j);

      if(j != (dataLine.size() - 1)) {
        fs << ",";
      } else {
        fs << endl;
      } 
    }   
  }

  fs.close();

  return 0;
}
