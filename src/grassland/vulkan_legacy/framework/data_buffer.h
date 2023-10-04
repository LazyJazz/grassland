#pragma once
#include <grassland/vulkan_legacy/framework/core.h>

namespace grassland::vulkan_legacy::framework {
class DataBuffer {
 public:
  explicit DataBuffer(Core *core);
  virtual ~DataBuffer() = default;

  [[nodiscard]] virtual Buffer *GetBuffer(int frame_index) const = 0;
  [[nodiscard]] virtual VkDeviceSize BufferSize() const = 0;
  virtual void Sync(int frame_index) = 0;

  [[nodiscard]] Core *GetCore() const;

 protected:
  Core *core_{nullptr};
};
}  // namespace grassland::vulkan_legacy::framework
