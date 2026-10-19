# TileLang to four-hart RVV vector addition

This demo lowers a TileLang float32 tile through the existing CPU backend, emits
C++, compiles it to RV32/RVV with Clang/LLVM, and executes four independent tiles
using Pokedex's `run-gpgpu` subcommand. Each hart processes one block. By default,
each block has 257 elements, exercising a tile size that is not a multiple of the
vector length.

This is an executable prototype, not a registered GPGPU backend or JIT adapter.
No TileLang compiler C++ or bundled TVM changes are required. It establishes the
code-generation and execution path before introducing a backend manifest or SDK.

## Prerequisites

- A working local TileLang build and its Python dependencies.
- Clang/Clang++, LLD, and `llvm-objdump` with RISC-V support.
- The sibling `../T1-thread/pokedex` checkout with the `run-gpgpu` extension.
- A Pokedex C++ model library with the same ABI as that checkout.

From the TileLang repository root, build the Rust runner:

```bash
cargo build --manifest-path ../T1-thread/pokedex/simulator/Cargo.toml \
  --target-dir "$PWD/build/pokedex-simulator"
```

The existing model library can be supplied directly. To build it from source,
use the T1 repository's `pokedex.model` Nix attribute, or configure Meson with
the Berkeley SoftFloat installation used by Pokedex:

```bash
meson setup build/pokedex-model ../T1-thread/pokedex/model_cpp \
  --buildtype=release \
  -Dpokedex_include_dir="$(realpath ../T1-thread/pokedex/simulator/include)" \
  -Dsoftfloat_include_dir=/path/to/softfloat/include \
  -Dsoftfloat_lib_dir=/path/to/softfloat/lib
ninja -C build/pokedex-model
```

Meson needs PyYAML in its selected Python interpreter, and the SoftFloat library
directory must contain `libsoftfloat.a` or a linkable equivalent.

## Run

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python examples/gpgpu/vector_add.py \
  --pokedex build/pokedex-simulator/debug/pokedex \
  --model-lib build/pokedex-model/libpokedex_model.so
```

The script checks that the kernel object contains RVV vector loads, addition,
stores and vector-length setup. Pokedex then executes the ELF. Each hart computes
its tile, publishes completion in shared memory, and waits. Hart 0 checks all
four outputs and their guard cells using scalar floating-point code before
reporting success through HTIF. Inputs are small binary fractions, allowing
exact float32 comparison. The verifier is compiled with vectorization disabled.

Expected final output:

```text
run-gpgpu: HTIF exit=0, harts=4, instructions=[...]
PASS: 4 harts × 257 float32 elements; RVV verified; artifacts in ...
```

Use `--vlen 128 --block-size 65` to exercise another vector length and tail.
`--build-dir`, `--clang`, and `--objdump` select artifact paths and tools.

`build/gpgpu-demo/` contains the original generated source, freestanding source,
lowered TIR, kernel disassembly, objects, ELF and a hart-tagged execution trace.

## Prototype boundaries

- The TileLang function describes one tile; the guest harness maps `mhartid` to
  a tile's buffer addresses. Grid scheduling is not yet represented in TileLang.
- TileLang's fixed-width CPU vectorization is disabled, leaving scalar C++ loops
  for LLVM's RVV auto-vectorizer.
- The demo replaces the CPU codegen's `tl_templates/cpp/common.h` include with
  `stdint.h`. This specific scalar float32 kernel needs no CPU runtime helpers.
  This narrowly scoped adaptation avoids needing a RISC-V C++ standard library;
  it is not a general-purpose freestanding code generator.
- The guest uses shared global SRAM and separate 8 KiB stacks. It does not use
  scratchpad memory or model its bandwidth, allocation or isolation.
- Four harts are interleaved instruction by instruction. Results validate
  functional execution, not hardware timing, SMT throughput or memory ordering.
- There is no PyTorch device integration, host/device transfer SDK, kernel cache,
  dynamic launch API, multi-SM scheduling, or performance benchmark yet.
