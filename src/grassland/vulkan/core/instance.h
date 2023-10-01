#pragma once

#include <vulkan/vulkan.h>

#include "grassland/vulkan/core/instance_procedures.h"
#include "grassland/vulkan/core/instance_settings.h"
#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan {

class PhysicalDevice;

class Instance {
 public:
  Instance();
  explicit Instance(InstanceSettings settings);
  ~Instance();

  [[nodiscard]] VkInstance Handle() const;

  void EnumeratePhysicalDevices(
      std::vector<PhysicalDevice> &physical_devices) const;

  [[nodiscard]] const InstanceSettings &Settings() const;
  InstanceProcedures &Procedures() {
    return instance_procedures_;
  }

 private:
  InstanceSettings settings_;
  VkInstance instance_{};
  VkDebugUtilsMessengerEXT debug_messenger_{};
  InstanceProcedures instance_procedures_{};
};
}  // namespace grassland::vulkan
   // Include Vulkan header and define a Vulkan Instance class
