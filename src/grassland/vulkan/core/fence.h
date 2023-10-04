#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
class Fence {
 public:
  explicit Fence(Device *device);
  ~Fence();

  [[nodiscard]] VkFence Handle() const;

 private:
  Device *device_{};
  VkFence fence_{};
};
}  // namespace grassland::vulkan
