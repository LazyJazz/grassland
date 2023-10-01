#pragma once
#include "string"

namespace grassland::util {
std::wstring U8StringToWideString(const std::string &str);
std::string WideStringToU8String(const std::wstring &wide_str);
std::string SizeToString(size_t size);
}  // namespace grassland::util
