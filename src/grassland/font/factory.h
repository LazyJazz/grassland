#pragma once

#include "fstream"
#include "grassland/font/mesh.h"
#include "map"
#include "string"

namespace grassland::font {
typedef wchar_t Char_T;
class Factory {
 public:
  explicit Factory(const char *font_file_path,
                   const char *font_cache_path = nullptr);
  ~Factory();
  const Mesh &GetChar(Char_T c);
  Mesh GetString(const std::wstring &wide_str);

 private:
  void LoadChar(Char_T c);
  FT_Library library_{};
  FT_Face face_{};
  std::map<Char_T, Mesh> loaded_fonts_;
  std::fstream cache_file_{};
  std::vector<size_t> cache_offset_{};
};
}  // namespace grassland::font
