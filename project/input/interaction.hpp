#pragma once
#include "../core/common/types.hpp"
#include "../physics/i_physics_solver.hpp"

struct InputState {
  bool mouse_down{false};
  f32  mouse_x{}, mouse_y{}; // NDC 或窗口坐标
  bool toggle_wind{false};
  bool reset_scene{false};
};

ExternalForces map_input_to_forces(const InputState& in, const ExternalForces& cur);

