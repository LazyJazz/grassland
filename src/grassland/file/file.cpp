#include <grassland/file/file.h>

namespace grassland::file {

std::string ReadFile(const char *file_path) {
  std::ifstream file_in(file_path, std::ios::ate);
  uint64_t file_size = uint64_t(file_in.tellg());
  file_in.seekg(std::ios::beg);
  std::string result;
  result.resize(file_size);
  file_in.read(result.data(), std::streamsize(file_size));
  return result;
}

std::vector<uint8_t> ReadFileBinary(const char *file_path) {
  std::ifstream file_in(file_path, std::ios::ate | std::ios::binary);
  uint64_t file_size = uint64_t(file_in.tellg());
  file_in.seekg(std::ios::beg);
  std::vector<uint8_t> result(file_size);
  file_in.read(reinterpret_cast<char *>(result.data()),
               std::streamsize(file_size));
  return result;
}

}  // namespace grassland::file
