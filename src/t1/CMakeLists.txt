# Target registration and freestanding source generation need no device SDK.
list(APPEND TILE_LANG_SRCS
  src/t1/target.cc
  src/t1/codegen/codegen.cc
)
