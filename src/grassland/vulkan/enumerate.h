#pragma once
#include "vulkan_util.h"

namespace grassland::vulkan {

template <class HandleType, class ValueType>
inline void GetEnumerateVector(HandleType handle,
                               void(enumerate)(HandleType,
                                               uint32_t *,
                                               ValueType *),
                               std::vector<ValueType> &result) {
  uint32_t count = 0;
  enumerate(handle, &count, nullptr);

  result.resize(count);
  enumerate(handle, &count, result.data());
}

template <class HandleType, class ValueType>
inline void GetEnumerateVector(HandleType handle,
                               VkResult(enumerate)(HandleType,
                                                   uint32_t *,
                                                   ValueType *),
                               std::vector<ValueType> &result) {
  uint32_t count = 0;
  GRASSLAND_VULKAN_CHECK(enumerate(handle, &count, nullptr));

  result.resize(count);
  GRASSLAND_VULKAN_CHECK(enumerate(handle, &count, result.data()));
}

template <class HandleType1, class HandleType2, class ValueType>
inline void GetEnumerateVector(
    HandleType1 handle1,
    HandleType2 handle2,
    void(enumerate)(HandleType1, HandleType2, uint32_t *, ValueType *),
    std::vector<ValueType> &result) {
  uint32_t count = 0;
  enumerate(handle1, handle2, &count, nullptr);

  result.resize(count);
  enumerate(handle1, handle2, &count, result.data());
}

template <class HandleType1, class HandleType2, class ValueType>
inline void GetEnumerateVector(
    HandleType1 handle1,
    HandleType2 handle2,
    VkResult(enumerate)(HandleType1, HandleType2, uint32_t *, ValueType *),
    std::vector<ValueType> &result) {
  uint32_t count = 0;
  GRASSLAND_VULKAN_CHECK(enumerate(handle1, handle2, &count, nullptr));

  result.resize(count);
  GRASSLAND_VULKAN_CHECK(enumerate(handle1, handle2, &count, result.data()));
}

template <class HandleType, class ValueType>
inline std::vector<ValueType> GetEnumerateVector(HandleType handle,
                                                 void(enumerate)(HandleType,
                                                                 uint32_t *,
                                                                 ValueType *)) {
  std::vector<ValueType> result;
  GetEnumerateVector(handle, enumerate, result);
  return result;
}

template <class HandleType, class ValueType>
inline std::vector<ValueType> GetEnumerateVector(
    HandleType handle,
    VkResult(enumerate)(HandleType, uint32_t *, ValueType *)) {
  std::vector<ValueType> result;
  GetEnumerateVector(handle, enumerate, result);
  return result;
}

template <class HandleType1, class HandleType2, class ValueType>
inline std::vector<ValueType> GetEnumerateVector(
    HandleType1 handle1,
    HandleType2 handle2,
    void(enumerate)(HandleType1, HandleType2, uint32_t *, ValueType *)) {
  std::vector<ValueType> result;
  GetEnumerateVector(handle1, handle2, enumerate, result);
  return result;
}

template <class HandleType1, class HandleType2, class ValueType>
inline std::vector<ValueType> GetEnumerateVector(
    HandleType1 handle1,
    HandleType2 handle2,
    VkResult(enumerate)(HandleType1, HandleType2, uint32_t *, ValueType *)) {
  std::vector<ValueType> result;
  GetEnumerateVector(handle1, handle2, enumerate, result);
  return result;
}

}  // namespace grassland::vulkan
