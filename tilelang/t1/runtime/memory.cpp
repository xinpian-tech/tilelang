// Freestanding helpers for memory intrinsics introduced by LLVM optimization.
#include <stddef.h>

extern "C" void *memcpy(void *destination, const void *source, size_t size) {
  auto *dst = static_cast<unsigned char *>(destination);
  const auto *src = static_cast<const unsigned char *>(source);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
  return destination;
}

extern "C" void *memset(void *destination, int value, size_t size) {
  auto *dst = static_cast<unsigned char *>(destination);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = static_cast<unsigned char>(value);
  }
  return destination;
}
