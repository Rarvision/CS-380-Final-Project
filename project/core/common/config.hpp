#pragma once
#include <string>
struct ClothPresetCfg {
  i32 nx{64}, ny{64};
  f32 spacing{0.02f};
  f32 mass{0.02f};
  f32 k_struct{8000.f}, k_shear{4000.f}, k_bend{0.f};
  f32 damping{0.01f};
  bool pin_top_edge{true};
};
struct RenderCfg { bool wireframe{false}; };
struct AppCfg {
  ClothPresetCfg cloth;
  RenderCfg render;
  f32 dt{1.0f/120.0f};
  bool fixed_dt{true};
};
AppCfg load_app_config(const std::string& path);
