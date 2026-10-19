from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen
from tvm import IRModule

from .target import llvm_target


def build(mod, target):
    if str(target.attrs["codegen"]) == "c":
        return global_func_device_codegen("target.build.tilelang_t1")(mod, target)
    internal_target = llvm_target(target)
    llvm_mod = IRModule({gv: func.with_attr("target", internal_target) for gv, func in mod.functions.items()})
    return global_func_device_codegen("target.build.llvm")(llvm_mod, internal_target)


T1_CODEGEN = DeviceCodegen("t1", build_without_compile=build)
