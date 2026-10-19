# T1: four-hart RVV backend

This example uses the registered `t1` target and `pokedex` execution backend.
Each of four SMT harts executes one block. `T.Parallel` becomes a scalar loop
inside that block; Clang auto-vectorizes suitable loops into RVV instructions.
No CUDA thread/lane mapping is involved.

## Build and run

From the TileLang repository root, activate the development environment and
build the native library. T1 registration and source codegen have no SDK
requirements and are always included; no `USE_T1` flag is needed.

```bash
source .venv/bin/activate
cmake -S . -B build -GNinja -DUSE_CUDA=OFF -DUSE_ROCM=OFF
cmake --build build -j4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Execution requires Clang with RV32/RVV support, `ld.lld`, CPU PyTorch, NumPy,
and Pokedex with the `run-gpgpu` memory-transfer options. Configure the Pokedex
executable and its matching model library:

```bash
export TILELANG_POKEDEX="$PWD/build/pokedex-simulator/debug/pokedex"
export POKEDEX_MODEL_LIB="$PWD/build/pokedex-model/libpokedex_model.so"
# Optional compiler override; otherwise clang++ is found on PATH.
export TILELANG_T1_CLANG=clang++

python examples/t1/vector_add.py
python examples/t1/vector_add.py --size 259 --vlen 128
```

These Pokedex paths refer to the local development build. See the sibling
[Pokedex instructions](../../../T1-thread/pokedex/README-gpgpu.md) for its build
and runner interface. Build the runner and model from matching source revisions.
The example runs two distinct input pairs through the same compiled kernel and
checks every result against PyTorch. VLEN defaults to 256 bits; target validation
accepts powers of two from 128 through 65536, subject to compiler/model support.
Only 128 and 256 have been exercised here.

## Python interface

```python
import tilelang
from examples.t1.vector_add import vector_add

kernel = tilelang.compile(
    vector_add(1025),
    target={"kind": "t1", "vlen": 256},
    execution_backend="pokedex",  # also selected by auto for target="t1"
    out_idx=-1,
)
c = kernel(a, b)  # contiguous CPU float32 tensors, shape (1025,)
```

Omit `out_idx` to pass all three tensors explicitly: `kernel(a, b, c)` updates
`c` and returns `None`. Arguments must have the declared shapes and cannot
share overlapping storage. Calls are synchronous. The adapter copies all
argument buffers back after successful completion, so explicit in-place
updates of an individual argument are supported. Auto-allocated output buffers
start at zero in simulated memory.

Inspect generated C++ using `kernel.get_kernel_source()` and the guest launcher
using `kernel.get_host_source()`. The ELF is at `kernel.adapter.libpath`; copy it
if it must outlive the kernel/cache. The generic disk cache stores that ELF under
its standard `kernel_lib.so` filename, but the file remains a bare-metal RV32 ELF.
`export_library()` is unsupported because this artifact is not a host shared
library. Source-only lowering (`tilelang.lower(func, target="t1")`) requires
neither the cross compiler nor the simulator.

Cache keys include the target, compile flags, T1 sources/runtime ABI, and
compiler/linker identities. Clang and LLD must remain available to validate the
cache key. Pokedex and its model are resolved at launch, allowing simulator
updates without recompiling unchanged guest code. Each launch starts fresh
simulated memory and hart state. Optional `TILELANG_CACHE_DIR` selects the cache
location.

## Initial capabilities

Supported programs have one top-level `T.Kernel(4)` launch, static nonempty
contiguous FP32 global buffers, scalar arithmetic/comparisons/conditionals, and
serial or parallel elementwise loops. Index arithmetic uses 32-bit integers.
The dialect is `import tilelang.t1.language as T`. Tensor shapes must be positive;
all arguments together must fit the 128 MiB transfer arena.

Unsupported features include SIMT `threads=`, vector IR, shared/local allocation,
atomics, tile operations (`T.copy`, GEMM, reductions), scalar kernel arguments,
dynamic shapes, multiple launches, other dtypes, min/max and floor division/modulo
expressions, and library math calls. Unsupported constructs fail during lowering.
Each block must access disjoint output elements; no cross-block synchronization
or race analysis is provided. Streams, profiling, and autotuning are unsupported.

The runtime assigns block ID = hart ID, gives each hart an 8 KiB stack, and waits
for all four harts before reporting HTIF success. A trap reports failure. Program
segments occupy `[0x80000000, 0x90000000)`; tensors are packed at 64-byte-aligned
addresses in `[0x90000000, 0x98000000)`. Transfers cannot overlap ELF segments or
reserved stacks. This is a functional simulator: it does not model SMT timing,
weak memory ordering, or scratchpad behavior.

A launch has a 60-second wall-time limit and a default aggregate instruction
budget of 10,000,000, including completion polling. Override the latter with
`TILELANG_POKEDEX_MAX_INSTRUCTIONS`. Failure or timeout raises a Python exception
and leaves caller tensors unchanged.

## Validation

With the runtime environment above:

```bash
python -m pytest -q testing/python/t1
python ../T1-thread/pokedex/tests/test_run_gpgpu.py \
  --pokedex "$TILELANG_POKEDEX" --model-lib "$POKEDEX_MODEL_LIB"
```

The execution tests need `llvm-objdump` to check that vector-add contains RVV
loads, additions, and stores. Without simulator environment variables, the
compiler tests run and execution tests skip.
