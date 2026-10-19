#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {
TVM_REGISTER_TARGET_KIND("t1", kDLExtDev)
    .add_attr_option<int64_t>("vlen", ffi::reflection::DefaultValue(256))
    .add_attr_option<bool>("supports_kernel_status_return",
                           ffi::reflection::DefaultValue(false))
    .set_default_keys({"t1"});
} // namespace tvm
