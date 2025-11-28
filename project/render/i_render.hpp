#pragma once
#include <cstddef>
#include "../core/common/types.hpp"
struct RenderView {
  void* device_positions{};
  void* device_normals{};
  void* device_indices{};
  u32   n_indices{};
};
class IRenderer {
public:
  virtual ~IRenderer() = default;
  virtual bool init(void* window_handle) = 0; // GLFWwindow*
  virtual void upload_mesh_views(const RenderView& v) = 0;
  virtual void draw_frame() = 0;
  virtual void resize(int w, int h) = 0;
};

