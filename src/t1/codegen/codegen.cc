#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

#include "backend/common/codegen/codegen_c_line_directives.h"
#include "support/check.h"

namespace tvm::codegen {

// T1 emits scalar freestanding C++; Clang performs RVV vectorization. The
// pipeline rejects vector IR and runtime-dependent operations before codegen.
class CodeGenT1 final : public CodeGenCWithLineDirectives {
public:
  void Initialize() {
    Init(false);
    decl_stream << "#include <stdint.h>\n#include <stddef.h>\n";
  }

  void PrintFuncPrefix(std::ostream &os) final { os << "extern \"C\" "; }
};

ffi::Module BuildT1(const IRModule &mod, const Target &target) {
  ICHECK_EQ(mod->functions.size(), 1) << "T1 currently requires one kernel";
  CodeGenT1 cg;
  cg.Initialize();
  ffi::Array<ffi::String> names;
  for (const auto &[gvar, base_func] : mod->functions) {
    tirx::PrimFunc func = Downcast<tirx::PrimFunc>(base_func);
    cg.DeclareFunction(gvar, func);
    cg.AddFunction(gvar, func);
    names.push_back(cg.GetFunctionName(gvar));
  }
  return CSourceModuleCreate(cg.Finish(), "c", names);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("target.build.tilelang_t1", BuildT1);
}
} // namespace tvm::codegen
