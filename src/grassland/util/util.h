#pragma once
#include <grassland/util/casting.h>
#include <grassland/util/logging.h>
#include <grassland/util/string_convert.h>

#define GRASSLAND_CANNOT_COPY(ClassName)            \
  ClassName(const ClassName &) = delete;            \
  ClassName(ClassName &&) = delete;                 \
  ClassName &operator=(const ClassName &) = delete; \
  ClassName &operator=(ClassName &&) = delete;\
