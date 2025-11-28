#pragma once
#include <glm/glm.hpp>
struct Camera {
  glm::mat4 view{}, proj{};
  void set_perspective(float fovy_deg, float aspect, float znear, float zfar);
  void orbit(float dx, float dy);
  void dolly(float dz);
};
