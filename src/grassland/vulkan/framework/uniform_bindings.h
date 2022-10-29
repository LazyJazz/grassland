#pragma once
#include <grassland/vulkan/framework/core.h>

namespace grassland::vulkan::framework {
class UniformBinding {
 public:
  virtual void UpdateDescriptorSet() = 0;

 private:
};

class UniformBindingBuffer : public UniformBinding {
 public:
 private:
};

class UniformBindingTextureSampler : public UniformBinding {
 public:
 private:
};

class UniformBindingTextureSamplers : public UniformBinding {
 public:
 private:
};

class UniformBindingStorageTexture : public UniformBinding {
 public:
 private:
};

class UniformBindingStorageTextures : public UniformBinding {
 public:
 private:
};
}  // namespace grassland::vulkan::framework
