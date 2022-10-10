#include <grassland/util/logging.h>
#include <grassland/vulkan/descriptor_set_layout.h>

namespace grassland::vulkan {

DescriptorSetLayout::DescriptorSetLayout(
    Device *device,
    const helper::DescriptorSetLayoutBindings &layout_bindings)
    : handle_{} {
  device_ = device;
  auto &bindings = layout_bindings.GetDescriptorSetLayoutBinding();
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = bindings.size();
  layoutInfo.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(device_->GetHandle(), &layoutInfo, nullptr,
                                  &handle_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create descriptor set layout!");
  }
}

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(device_->GetHandle(), GetHandle(), nullptr);
}
}  // namespace grassland::vulkan
