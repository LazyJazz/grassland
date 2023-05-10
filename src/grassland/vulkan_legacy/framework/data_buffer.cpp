#include <grassland/vulkan_legacy/framework/data_buffer.h>

namespace grassland::vulkan_legacy::framework {

DataBuffer::DataBuffer(Core *core) {
  core_ = core;
}

Core *DataBuffer::GetCore() const {
  return core_;
}
}  // namespace grassland::vulkan_legacy::framework
