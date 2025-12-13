#pragma once
struct GLFWwindow;
class VulkanRenderer;
struct UIState;

namespace ui {

void init(GLFWwindow* window, VulkanRenderer* renderer);
void shutdown();

void new_frame();
void draw_panel(UIState& state);
void end_frame(VulkanRenderer* renderer);

} // namespace ui
