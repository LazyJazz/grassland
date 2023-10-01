#include "grassland/vulkan/core/instance_procedures.h"

namespace grassland::vulkan {

namespace {
template <class FuncTy>
FuncTy GetProcedure(VkInstance instance, const char *function_name) {
  auto func = (FuncTy)vkGetInstanceProcAddr(instance, function_name);
  return func;
};
}  // namespace

#define GET_PROCEDURE(instance, function_name)                             \
  function_name##_ = grassland::vulkan::GetProcedure<PFN_##function_name>( \
      instance, #function_name)

void InstanceProcedures::GetFunctionPointers(VkInstance instance) {
  GET_PROCEDURE(instance, vkCreateDebugUtilsMessengerEXT);
  GET_PROCEDURE(instance, vkDestroyDebugUtilsMessengerEXT);
}

VkResult InstanceProcedures::vkCreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pMessenger) {
  return vkCreateDebugUtilsMessengerEXT_(instance, pCreateInfo, pAllocator,
                                         pMessenger);
}

void InstanceProcedures::vkDestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks *pAllocator) {
  vkDestroyDebugUtilsMessengerEXT_(instance, messenger, pAllocator);
}

}  // namespace grassland::vulkan
