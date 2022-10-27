#include <grassland/util/util.h>
#include <grassland/vulkan/framework/core.h>

namespace grassland::vulkan::framework {

Core::Core(const CoreSettings &core_settings) {
  core_settings_ = core_settings;
  if (core_settings_.has_window) {
    if (!glfwInit()) {
      LAND_ERROR("[Vulkan] GLFW initialization failed.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window_ = glfwCreateWindow(int(core_settings_.window_width),
                               int(core_settings_.window_height),
                               core_settings_.window_title, nullptr, nullptr);
    if (!window_) {
      LAND_ERROR("[Vulkan] GLFW create window failed.");
    }
  }

  instance_ = std::make_unique<Instance>(core_settings_.has_window,
                                         core_settings_.validation_layer);

  if (core_settings_.has_window) {
    surface_ = std::make_unique<Surface>(instance_.get(), window_);
  }

  physical_device_ = std::make_unique<PhysicalDevice>(PickPhysicalDevice(
      instance_.get(), [&core_settings](PhysicalDevice physical_device) {
        if (core_settings.raytracing_pipeline_required &&
            !physical_device.HasRayTracingPipeline()) {
          return -1;
        }
        int score = 0;
        if (core_settings.has_window &&
            !physical_device.HasPresentationSupport()) {
          return -1;
        }
        score += int(physical_device.DeviceMemorySize() >> 20);
        if (physical_device.IsDiscreteGPU()) {
          score *= 2;
        }
        return score;
      }));

  physical_device_->PrintDeviceProperties();

  if (!physical_device_->GetHandle()) {
    LAND_ERROR("[Vulkan] failed to find available device.");
  }

  device_ = std::make_unique<Device>(
      physical_device_.get(),
      core_settings_.has_window ? surface_.get() : nullptr,
      core_settings_.raytracing_pipeline_required
          ? std::vector<
                const char *>{VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                              VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                              VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}
          : std::vector<const char *>{},
      core_settings_.validation_layer);

  if (core_settings_.has_window) {
    swapchain_ = std::make_unique<Swapchain>(device_.get(), window_);
  }

  command_pool_ = std::make_unique<CommandPool>(device_.get());
  command_buffers_.clear();
  for (int i = 0; i < core_settings_.frames_in_flight; i++) {
    command_buffers_.push_back(
        std::make_unique<CommandBuffer>(command_pool_.get()));
  }

  for (int i = 0; i < core_settings_.frames_in_flight; i++) {
    in_flight_fences_.push_back(std::make_unique<Fence>(device_.get()));
    image_available_semaphores_.push_back(
        std::make_unique<Semaphore>(device_.get()));
    render_finish_semaphores_.push_back(
        std::make_unique<Semaphore>(device_.get()));
  }
}

Core::~Core() {
  device_->WaitIdle();
  render_finish_semaphores_.clear();
  image_available_semaphores_.clear();
  in_flight_fences_.clear();
  command_buffers_.clear();
  command_pool_.reset();
  swapchain_.reset();
  device_.reset();
  physical_device_.reset();
  if (core_settings_.has_window) {
    surface_.reset();
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
  instance_.reset();
}
const CoreSettings &Core::GetCoreSettings() const {
  return core_settings_;
}
Instance *Core::GetInstance() const {
  return instance_.get();
}
PhysicalDevice *Core::GetPhysicalDevice() const {
  return physical_device_.get();
}
Device *Core::GetDevice() const {
  return device_.get();
}
Surface *Core::GetSurface() const {
  return surface_.get();
}
Swapchain *Core::GetSwapchain() const {
  return swapchain_.get();
}
CommandPool *Core::GetCommandPool() const {
  return command_pool_.get();
}
GLFWwindow *Core::GetWindow() const {
  return window_;
}

}  // namespace grassland::vulkan::framework
