#ifndef _FETCH_CSV_DATA_HPP_
#define _FETCH_CSV_DATA_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>

using namespace std;

namespace utils 
{
  class FetchCSVData
  {
  public:
    FetchCSVData(string filename);
    vector<vector<double> > execute();

  private:
    vector<vector<double> > data;
  };
}

#endif
