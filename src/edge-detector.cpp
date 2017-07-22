#include <iostream>
#include <vector>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../include/json.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv) {
  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  auto configJson = json::parse(str);

  int width;
  int height;

  return 0;
}
