cmake_minimum_required(VERSION 3.1)
project(JSON VERSION 3.8.0 LANGUAGES CXX)

add_library(json INTERFACE)
target_include_directories(json INTERFACE include)
add_library(json::json ALIAS json)
