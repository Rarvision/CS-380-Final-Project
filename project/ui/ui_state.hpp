#pragma once
#include <glm/glm.hpp>

// physical presets for a single fabric material
struct ClothMaterialPreset {
    float mass;
    float k_struct;
    float k_shear;
    float k_bend;
};

// UI params
struct UIState {
    // wind related
    bool  wind_enabled       = false;
    float wind_base_strength = 25.0f;

    // cloth physics (currrent)
    float cloth_mass       = 0.02f;
    float cloth_k_struct   = 10000.0f;
    float cloth_k_shear    = 24000.0f;
    float cloth_k_bend     = 16000.0f;

    // cloth physics
    float prev_cloth_mass     = cloth_mass;
    float prev_cloth_k_struct = cloth_k_struct;
    float prev_cloth_k_shear  = cloth_k_shear;
    float prev_cloth_k_bend   = cloth_k_bend;

    // pin control
    bool  pin_top_edge       = true;
    bool  prev_pin_top_edge  = true;
    bool  request_reset_hang = false;

    // material selection 0: Elastane 1: Cotton 2: PVC
    int   current_material_index = 0;  // 0,1,2
    int   prev_material_index    = 0;

    // default preset of materials
    ClothMaterialPreset elastane_preset   { 0.015f,  8000.0f, 18000.0f, 12000.0f };
    ClothMaterialPreset cotton_preset  { 0.030f, 16000.0f, 26000.0f, 20000.0f };
    ClothMaterialPreset pvc_preset{ 0.020f, 12000.0f, 30000.0f, 26000.0f };

    const char* material_names[3] = {
        "Elastane",
        "Cotton",
        "PVC"
    };

    bool cloth_params_dirty = false;
};

static void apply_material_preset(UIState& ui)
{
    switch (ui.current_material_index) {
    case 0: // elastane：light, soft
        ui.cloth_mass     = 0.015f;
        ui.cloth_k_struct = 22000.0f;
        ui.cloth_k_shear  = 12000.0f;
        ui.cloth_k_bend   = 3000.0f;
        break;

    case 1: // cotton
        ui.cloth_mass     = 0.050f;
        ui.cloth_k_struct = 45000.0f;
        ui.cloth_k_shear  = 28000.0f;
        ui.cloth_k_bend   = 9000.0f;
        break;

    case 2: // PVC cloth
        ui.cloth_mass     = 0.025f;
        ui.cloth_k_struct = 26000.0f;
        ui.cloth_k_shear  = 22000.0f;
        ui.cloth_k_bend   = 14000.0f; 
        ui.wind_base_strength = 18.0f;
        break;

    default:
        break;
    }

    ui.cloth_params_dirty = true;
}