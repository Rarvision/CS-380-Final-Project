#pragma once
#include "../core/cloth/cloth_model.hpp"
#include "../core/common/types.hpp"

enum class SceneType { HangingCloth };

struct Scene {
  SceneType type{SceneType::HangingCloth};
  ClothModel cloth;
  ExternalForces forces;
};
