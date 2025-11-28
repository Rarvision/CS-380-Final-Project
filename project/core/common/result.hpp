#pragma once
#include <string>
template<typename T> struct Result {
  T value{};
  bool ok{true};
  std::string err{};
  static Result<T> failure(std::string e){ return {{}, false, std::move(e)}; }
};
