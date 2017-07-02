#include "../../include/utils/Misc.hpp"

vector<vector<double> > utils::Misc::fetchCSVData(string filename) {
  vector<vector<double> > data;

  ifstream infile(filename);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stof(tok));
    }

    data.push_back(dRow);
  }

  return data;
}
