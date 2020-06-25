cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(iml LANGUAGES CXX VERSION 1.2)

add_library(iml INTERFACE)

target_include_directories(iml INTERFACE include/)

add_library(iml::iml ALIAS iml)
