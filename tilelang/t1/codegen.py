from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen

T1_CODEGEN = DeviceCodegen("t1", build_without_compile=global_func_device_codegen("target.build.tilelang_t1"))
