#include "grassland/vulkan/core/instance_procedures.h"

namespace grassland::vulkan {

namespace {
template <class FuncTy>
FuncTy GetProcedure(VkInstance instance, const char *function_name) {
  auto func = (FuncTy)vkGetInstanceProcAddr(instance, function_name);
  if (!func) {
    LAND_WARN("Failed to load instance function: {}", function_name);
  }
  return func;
};
}  // namespace

#define GET_PROCEDURE(instance, function_name)                          \
  function_name = grassland::vulkan::GetProcedure<PFN_##function_name>( \
      instance, #function_name)

void InstanceProcedures::Initialize(VkInstance instance) {
  GET_PROCEDURE(instance, vkCreateDebugUtilsMessengerEXT);
  GET_PROCEDURE(instance, vkDestroyDebugUtilsMessengerEXT);
  GET_PROCEDURE(instance, vkSetDebugUtilsObjectNameEXT);
}

}  // namespace grassland::vulkan
