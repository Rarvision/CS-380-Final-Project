#pragma once
#include <glm/glm.hpp>

// 单个布材质的物理预设
struct ClothMaterialPreset {
    float mass;
    float k_struct;
    float k_shear;
    float k_bend;
};

// 所有 UI 可调参数统一放在这里
struct UIState {
    // -------- 1) 风场参数 --------
    bool  wind_enabled       = false;
    float wind_base_strength = 25.0f;

    // -------- 2) 布的物理属性（当前值） --------
    float cloth_mass       = 0.02f;
    float cloth_k_struct   = 10000.0f;
    float cloth_k_shear    = 24000.0f;
    float cloth_k_bend     = 16000.0f;

    // 用于之后做“参数改变 → 重建布”时判断
    float prev_cloth_mass     = cloth_mass;
    float prev_cloth_k_struct = cloth_k_struct;
    float prev_cloth_k_shear  = cloth_k_shear;
    float prev_cloth_k_bend   = cloth_k_bend;

    // -------- 3) 悬挂 / 掉落 控制 --------
    bool  pin_top_edge       = true;   // 当前是否应该固定上边
    bool  prev_pin_top_edge  = true;
    bool  request_reset_hang = false;  // 用户点击“恢复悬挂”的一次性请求

    // -------- 4) 材质选择（0 丝绸 / 1 硬布 / 2 塑料） --------
    int   current_material_index = 0;  // 0,1,2
    int   prev_material_index    = 0;

    // 三种材质的物理预设（之后可以微调）
    ClothMaterialPreset silk_preset   { 0.015f,  8000.0f, 18000.0f, 12000.0f };
    ClothMaterialPreset stiff_preset  { 0.030f, 16000.0f, 26000.0f, 20000.0f };
    ClothMaterialPreset plastic_preset{ 0.020f, 12000.0f, 30000.0f, 26000.0f };

    const char* material_names[3] = {
        "Silk",
        "Stiff Cloth",
        "Plastic"
    };

    bool cloth_params_dirty = false;
};

// 下拉菜单切换材质时，把预设同步到 cloth_* 上
// inline void apply_material_preset(UIState& ui)
// {
//     const ClothMaterialPreset* preset = nullptr;
//     switch (ui.current_material_index) {
//     case 0: preset = &ui.silk_preset;    break;
//     case 1: preset = &ui.stiff_preset;   break;
//     case 2: preset = &ui.plastic_preset; break;
//     default: return;
//     }

//     ui.cloth_mass     = preset->mass;
//     ui.cloth_k_struct = preset->k_struct;
//     ui.cloth_k_shear  = preset->k_shear;
//     ui.cloth_k_bend   = preset->k_bend;
// }

static void apply_material_preset(UIState& ui)
{
    switch (ui.current_material_index) {
    case 0: // Silk：轻、柔软
        ui.cloth_mass     = 0.015f;    // 比较轻
        ui.cloth_k_struct = 22000.0f;  // 结构刚度中等偏低
        ui.cloth_k_shear  = 12000.0f;  // 剪切偏低 -> 易产生柔和褶皱
        ui.cloth_k_bend   = 3000.0f;   // 抗弯小 -> 很软
        break;

    case 1: // Heavy cloth：厚布/窗帘
        ui.cloth_mass     = 0.050f;    // 更重
        ui.cloth_k_struct = 45000.0f;  // 结构刚度高
        ui.cloth_k_shear  = 28000.0f;
        ui.cloth_k_bend   = 9000.0f;   // 折痕比较钝
        break;

    case 2: // Plastic：塑料布，拉不太开 & 折痕比较硬
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