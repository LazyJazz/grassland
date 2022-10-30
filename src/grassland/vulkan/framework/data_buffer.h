#pragma once
#include <grassland/vulkan/framework/core.h>

namespace grassland::vulkan::framework {
class DataBuffer {
 public:
  explicit DataBuffer(Core *core);
  virtual ~DataBuffer() = default;
  [[nodiscard]] virtual Buffer *GetBuffer(int image_index) const = 0;
  [[nodiscard]] Core *GetCore() const;

 protected:
  Core *core_{nullptr};
};
}  // namespace grassland::vulkan::framework
