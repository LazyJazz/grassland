#pragma once

#define VK_HANDLE(type)    \
  type handle_;            \
                           \
 public:                   \
  type GetHandle() const { \
    return handle_;        \
  }                        \
                           \
 private:

namespace grassland::vulkan {}
