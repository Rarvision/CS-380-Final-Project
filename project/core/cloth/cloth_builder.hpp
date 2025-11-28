// core/cloth/cloth_builder.hpp
#pragma once
#include "cloth_model.hpp"

ClothModel build_regular_grid(const ClothBuildParams& p);
void tag_pins_top_edge(ClothModel& c);
void compute_springs(ClothModel& c, f32 k_struct, f32 k_shear, f32 k_bend);
