#include "grassland/util/string_convert.h"

#include "fmt/format.h"

namespace grassland::util {

std::wstring U8StringToWideString(const std::string &str) {
  wchar_t wc;
  std::wstring result;
  int tail = 0;
  for (auto c : str) {
    if (c & 0b10000000) {
      int prefix_length = 1;
      uint8_t test_bit = 0b01000000;
      while (uint8_t(c) & test_bit) {
        prefix_length++;
        test_bit >>= 1;
      }
      int valid_bit = 7 - prefix_length;
      wchar_t valid_value = c & ((1 << valid_bit) - 1);
      if (prefix_length == 1) {
        tail--;
        wc <<= valid_bit;
        wc |= valid_value;
        if (!tail) {
          result += wc;
        }
      } else {
        tail = prefix_length - 1;
        wc = valid_value;
      }
    } else {
      result += wchar_t(c);
    }
  }
  return result;
}

std::string WideStringToU8String(const std::wstring &wide_str) {
  std::string str;
  for (auto wc : wide_str) {
    if ((wc & 0b1111111) == wc) {
      str += char(wc);
    } else if ((wc & 0b11111111111) == wc) {
      str += char(0b11000000 | ((wc >> 6) & 0b00011111));
      str += char(0b10000000 | (wc & 0b00111111));
    } else {
      str += char(0b11100000 | ((wc >> 12) & 0b00001111));
      str += char(0b10000000 | ((wc >> 6) & 0b00111111));
      str += char(0b10000000 | (wc & 0b00111111));
    }
  }
  return str;
}

std::string SizeToString(size_t size) {
  // convert size to string, alternate unit from byte to PB, use
  // double-precision, round to 2 decimal places
  std::string unit[] = {"B", "KB", "MB", "GB", "TB", "PB"};
  int i = 0;
  double size_d = size;
  while (size_d >= 1024 && i < 5) {
    size_d /= 1024;
    i++;
  }
  return fmt::format("{:.2f} {}", size_d, unit[i]);
}

std::string PCIVendorIDToName(uint32_t vendor_id) {
  switch (vendor_id) {
    case 0x1002:
      return "AMD (ATI)";
    case 0x1022:
      return "AMD";
    case 0x10DE:
      return "NVIDIA";
    case 0x8086:
      return "Intel";
    case 0x106B:
      return "Apple";
    case 0x5143:
    case 0x17CB:
      return "Qualcomm";
    case 0x19E5:
      return "HUAWEI";
    case 0x13B5:
      return "ARM";
    default:
      return fmt::format("Unknown vendor id: 0x{:04X}", vendor_id);
  }
}

}  // namespace grassland::util
