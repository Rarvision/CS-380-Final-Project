#pragma once
struct GLFWwindow;
struct Window {
  GLFWwindow* handle{nullptr};
  bool create(int w, int h, const char* title);
  void poll();
  bool should_close() const;
  void destroy();
};
