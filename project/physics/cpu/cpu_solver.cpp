class CpuSolver : public IPhysicsSolver {
public:
  void upload(const ClothModel&) override;
  void step(f32 dt, const ExternalForces&) override;
  void download(ClothModel&) override;
  void* get_device_position_buffer() override; // 返回空 / stub
  void* get_device_normal_buffer() override;
  void* get_device_index_buffer() override;
};
