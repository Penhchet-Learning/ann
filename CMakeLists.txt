cmake_minimum_required(VERSION 3.0)
project(openann)

set(CMAKE_BUILD_TYPE          Debug)
set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -pthread -std=c++14")

# main app
add_executable(
  openann
  src/main.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/NeuralNetwork/NeuralNetwork.cpp
  src/NeuralNetwork/backPropagation.cpp
  src/NeuralNetwork/feedForward.cpp
  src/utils/MultiplyMatrix.cpp
  src/utils/FetchCSVData.cpp
)
