#include <grassland/logging  //logging.h>

namespace grassland::log {
void assert_handler(bool result, const char *code) {
  if (!result) {
    LAND_ERROR("Grassland assert: {}", code);
  }
}
}  // namespace grassland::log
