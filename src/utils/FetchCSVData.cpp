#include "../../include/utils/FetchCSVData.hpp"

utils::FetchCSVData::FetchCSVData(string filename) {
  ifstream infile(filename);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stod(tok));
    }

    data.push_back(dRow);
  }
}

vector<vector<double> > utils::FetchCSVData::execute() {
  return this->data;
}
