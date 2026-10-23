"""Compiler tests need no device SDK; execution tests opt in via simulator paths."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

import tilelang
from tilelang.backend import create_backend_context
import tilelang.t1.language as T
from tilelang.t1 import toolchain
from examples.t1.vector_add import vector_add
from tvm import tirx
from tvm.target.codegen import llvm_version_major


CODEGENS = ["c", pytest.param("llvm", marks=pytest.mark.skipif(llvm_version_major(allow_none=True) is None, reason="LLVM disabled"))]


def lower(func):
    return tilelang.lower(func, target="t1")


def test_context_and_source_only(monkeypatch):
    monkeypatch.setenv("PATH", "")
    context = create_backend_context("t1", None, "auto")
    assert context.module.name == "t1"
    assert context.execution_backend.name == "pokedex"
    assert int(context.target.attrs["vlen"]) == 256
    artifact = lower(vector_add())
    assert "#include <stdint.h>" in artifact.kernel_source
    assert 'extern "C" void main(float*' in artifact.kernel_source
    assert "int32_t bx" in artifact.kernel_source
    assert "tl_templates" not in artifact.kernel_source
    function = next(iter(artifact.device_mod.functions.values()))
    assert function.attrs["target"].kind.name == "t1"
    assert int(function.attrs["t1.grid"]) == 4


@pytest.mark.parametrize("vlen", [0, 64, 192, 131072])
def test_invalid_vlen(vlen):
    with pytest.raises(ValueError, match="vlen must be a power"):
        create_backend_context({"kind": "t1", "vlen": vlen}, None, "auto")


def test_invalid_codegen():
    with pytest.raises(ValueError, match="codegen must be"):
        create_backend_context({"kind": "t1", "codegen": "invalid"}, None, "auto")


def test_missing_llvm(monkeypatch):
    monkeypatch.setattr("tvm.target.codegen.llvm_version_major", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="USE_LLVM=ON"):
        tilelang.lower(vector_add(), target={"kind": "t1", "codegen": "llvm"})
    assert 'extern "C"' in lower(vector_add()).kernel_source


@pytest.mark.parametrize("vlen", [128, 256])
@pytest.mark.skipif(llvm_version_major(allow_none=True) is None, reason="LLVM disabled")
def test_explicit_vector_source(monkeypatch, vlen):
    monkeypatch.setenv("PATH", "")
    artifact = tilelang.lower(vector_add(), target={"kind": "t1", "codegen": "llvm", "vlen": vlen})
    assert f"fadd <{vlen // 32} x float>" in artifact.kernel_source
    assert "llvm.masked.store" in artifact.kernel_source
    assert 'target triple = "riscv32-unknown-elf"' in artifact.kernel_source
    function = next(iter(artifact.device_mod.functions.values()))
    assert function.attrs["target"].kind.name == "t1"
    assert int(function.attrs["t1.grid"]) == 4
    stores = []
    tirx.stmt_functor.post_order_visit(function.body, lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None)
    assert any(str(store.value.dtype) == f"float32x{vlen // 32}" for store in stores)


@pytest.mark.parametrize("codegen", CODEGENS)
def test_min_max_lowering(codegen):
    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.Parallel(8):
                B[bx * 8 + i] = T.max(T.min(A[bx * 8 + i], 1.0), -1.0)

    artifact = tilelang.lower(main, target={"kind": "t1", "codegen": codegen, "vlen": 256})
    assert artifact.kernel_source
    if codegen == "llvm":
        function = next(iter(artifact.device_mod.functions.values()))
        stores = []
        tirx.stmt_functor.post_order_visit(function.body, lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None)
        assert any(str(store.value.dtype) == "float32x8" for store in stores)


def standard_ir_kernel(kind):
    if kind == "integer":

        @T.prim_func
        def main(A: T.Tensor((36,), "float32"), B: T.Tensor((4, 36), "float32")):
            with T.Kernel(4) as bx:
                for i in T.Parallel(9):
                    j = bx * 9 + i
                    x = T.Cast("int32", A[j])
                    divisor = bx * 2 - 3
                    B[0, j] = T.Cast("float32", x // divisor)
                    B[1, j] = T.Cast("float32", x % divisor)
                    B[2, j] = T.Cast("float32", (T.Cast("int8", x) >> 2) & 31)
                    B[3, j] = T.reinterpret(T.reinterpret(A[j], "uint32") ^ T.uint32(2147483648), "float32")

    elif kind == "local":

        @T.prim_func
        def main(A: T.Tensor((36,), "float32"), B: T.Tensor((36,), "float32")):
            with T.Kernel(4) as bx:
                total = T.alloc_var("float32", init=0.0)
                scratch = T.alloc_local((9,), "float32")
                for i in T.serial(9):
                    scratch[i] = A[bx * 9 + i] * 2.0
                for i in T.serial(9):
                    total = total + scratch[i]
                for i in T.Parallel(9):
                    B[bx * 9 + i] = A[bx * 9 + i] * 2.0 + total

    else:

        @T.prim_func
        def main(A: T.Tensor((36,), "float32"), B: T.Tensor((36,), "float32")):
            with T.Kernel(4) as bx:
                for i in T.Parallel(9):
                    B[bx * 9 + i] = T.sqrt(T.clamp(T.abs(A[bx * 9 + i]), 0.0, 256.0))

    return main


STANDARD_IR_CASES = [(codegen, kind) for codegen in ("c", "llvm") for kind in ("integer", "local")]
STANDARD_IR_CASES.append(("llvm", "math"))


@pytest.mark.parametrize(("codegen", "kind"), STANDARD_IR_CASES)
def test_standard_ir_lowering(codegen, kind):
    if codegen == "llvm" and llvm_version_major(allow_none=True) is None:
        pytest.skip("LLVM disabled")
    artifact = tilelang.lower(standard_ir_kernel(kind), target={"kind": "t1", "codegen": codegen, "vlen": 256})
    if kind == "math":
        assert "llvm.sqrt.v8f32" in artifact.kernel_source
        assert "sqrtf" not in artifact.kernel_source


@pytest.mark.skipif(llvm_version_major(allow_none=True) is None, reason="LLVM disabled")
def test_exp_uses_llvm_intrinsic():
    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.Parallel(8):
                B[bx * 8 + i] = T.exp(A[bx * 8 + i])

    artifact = tilelang.lower(main, target={"kind": "t1", "codegen": "llvm", "vlen": 256})
    assert "llvm.exp.v8f32" in artifact.kernel_source
    assert "expf" not in artifact.kernel_source


@pytest.mark.skipif(llvm_version_major(allow_none=True) is None, reason="LLVM disabled")
def test_unsupported_vector_control_flow():
    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.Parallel(8):
                if A[bx * 8 + i] > 0:
                    B[bx * 8 + i] = A[bx * 8 + i]

    with pytest.raises(ValueError, match="could not explicitly vectorize"):
        tilelang.lower(main, target={"kind": "t1", "codegen": "llvm"})


def test_execution_compatibility():
    with pytest.raises(ValueError, match="tvm_ffi"):
        create_backend_context("t1", None, "tvm_ffi")
    with pytest.raises(ValueError, match="pokedex"):
        create_backend_context("c", None, "pokedex")


@pytest.mark.parametrize("blocks", [1, 3, 5, 8])
def test_invalid_grid(blocks):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(blocks) as bx:
            A[bx] = 1.0

    with pytest.raises(ValueError, match=r"T.Kernel\(4\)"):
        lower(main)


def test_explicit_threads_are_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'threads'"):

        @T.prim_func
        def main(A: T.Tensor((4,), "float32")):
            with T.Kernel(4, threads=4) as bx:
                A[bx] = 1.0


def test_autotuning_is_rejected():
    from tilelang.autotuner import AutoTuner

    with pytest.raises(NotImplementedError, match="functional simulator"):
        AutoTuner(vector_add, [{"n": 8}]).set_compile_args(target="t1")


@pytest.mark.parametrize("dtype", ["float16", "int32"])
def test_unsupported_tensor_dtype(dtype):
    @T.prim_func
    def main(A: T.Tensor((4,), dtype)):
        with T.Kernel(4) as bx:
            A[bx] = 1

    with pytest.raises(ValueError, match="global float32 buffers"):
        lower(main)


def test_shared_memory_is_rejected():
    @T.prim_func
    def main(A: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            shared = T.alloc_shared((4,), "float32")
            shared[bx] = A[bx]
            A[bx] = shared[bx]

    with pytest.raises(ValueError, match="allocations in scope.*are not implemented"):
        lower(main)


def test_atomic_is_rejected():
    @T.prim_func
    def main(A: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            T.atomic_add(A[bx], 1.0)

    with pytest.raises(ValueError, match="does not yet support operation"):
        lower(main)


def test_multiple_launches_are_rejected():
    @T.prim_func
    def main(A: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            A[bx] = 1.0
        with T.Kernel(4) as bx:
            A[bx] = 2.0

    with pytest.raises(ValueError, match="entire function body"):
        lower(main)


@pytest.fixture
def simulator():
    executable = os.environ.get("TILELANG_POKEDEX")
    model = os.environ.get("POKEDEX_MODEL_LIB")
    if not executable or not shutil.which(executable) or not model or not Path(model).is_file():
        pytest.skip("set TILELANG_POKEDEX and POKEDEX_MODEL_LIB for execution tests")
    return executable


@pytest.mark.parametrize(("codegen", "kind"), STANDARD_IR_CASES)
def test_standard_ir_execution(simulator, codegen, kind):
    if codegen == "llvm" and llvm_version_major(allow_none=True) is None:
        pytest.skip("LLVM disabled")
    flags = ["-fno-vectorize", "-fno-slp-vectorize"] if codegen == "llvm" else None
    kernel = tilelang.compile(
        standard_ir_kernel(kind), target={"kind": "t1", "codegen": codegen, "vlen": 256}, out_idx=-1, compile_flags=flags
    )
    a = torch.arange(-18, 18, dtype=torch.float32) * 17
    if kind == "integer":
        x = a.to(torch.int32)
        divisor = torch.tensor([-3, -1, 1, 3], dtype=torch.int32).repeat_interleave(9)
        expected = torch.stack(
            [
                torch.div(x, divisor, rounding_mode="floor").float(),
                torch.remainder(x, divisor).float(),
                ((x.to(torch.int8) >> 2) & 31).float(),
                -a,
            ]
        )
    elif kind == "local":
        doubled = a.reshape(4, 9) * 2
        expected = (doubled + doubled.sum(dim=1, keepdim=True)).reshape(36)
    else:
        expected = torch.sqrt(torch.clamp(torch.abs(a), 0, 256))
    torch.testing.assert_close(kernel(a), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize(("n", "vlen"), [(1025, 256), (259, 128), (3, 256)])
@pytest.mark.parametrize("codegen", CODEGENS)
def test_execution_repeated_inputs_and_tail(simulator, n, vlen, codegen):
    # Explicit TIR vectorization must survive even with Clang autovectorization disabled.
    flags = ["-fno-vectorize", "-fno-slp-vectorize"] if codegen == "llvm" else None
    kernel = tilelang.compile(vector_add(n), target={"kind": "t1", "vlen": vlen, "codegen": codegen}, out_idx=-1, compile_flags=flags)
    for seed in (7, 19):
        generator = torch.Generator().manual_seed(seed)
        a, b = (torch.randn(n, generator=generator) for _ in range(2))
        torch.testing.assert_close(kernel(a, b), a + b, rtol=0, atol=0)
    if n > 3:
        dump = subprocess.check_output(["llvm-objdump", "-d", kernel.adapter.libpath], text=True)
        assert re.search(r"\bvle32\.v\b", dump)
        assert re.search(r"\bvfadd\.vv\b", dump)
        assert re.search(r"\bvse32\.v\b", dump)


@pytest.mark.parametrize("codegen", CODEGENS)
def test_four_block_ids_and_explicit_outputs(simulator, codegen):
    @T.prim_func
    def main(A: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            A[bx] = T.Cast("float32", bx)

    kernel = tilelang.compile(main, target={"kind": "t1", "codegen": codegen})
    output = torch.full((4,), -1.0)
    assert kernel(output) is None
    torch.testing.assert_close(output, torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize("codegen", CODEGENS)
def test_masked_store_preserves_inactive_elements(simulator, codegen):
    @T.prim_func
    def main(A: T.Tensor((40,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.vectorized(8):
                index = bx * 8 + i
                if index < 31:
                    A[index] = A[index] + 1.0

    if codegen == "c":
        with pytest.raises(ValueError, match="explicit vector"):
            tilelang.lower(main, target="t1")
        return
    kernel = tilelang.compile(main, target={"kind": "t1", "codegen": codegen})
    a = torch.full((40,), 17.0)
    expected = a.clone()
    expected[:31] += 1
    kernel(a)
    torch.testing.assert_close(a, expected, rtol=0, atol=0)


def test_argument_and_failure_diagnostics(simulator, monkeypatch):
    kernel = tilelang.compile(vector_add(8), target="t1", out_idx=2)
    a, b = torch.randn(8), torch.randn(8)
    with pytest.raises(ValueError, match="expects 2"):
        kernel(a)
    with pytest.raises(ValueError, match="CPU float32"):
        kernel(a.double(), b)
    with pytest.raises(ValueError, match="contiguous tensor"):
        kernel(torch.randn(16)[::2], b)
    with pytest.raises(ValueError, match="must not overlap"):
        kernel(a, a)
    with pytest.raises(NotImplementedError, match="functional simulator"):
        kernel.get_profiler()
    with pytest.raises(NotImplementedError, match="bare-metal ELF"):
        kernel.export_library("unused.so")
    monkeypatch.setenv("TILELANG_POKEDEX_MAX_INSTRUCTIONS", "1")
    with pytest.raises(RuntimeError, match="instruction limit"):
        kernel(a, b)
    monkeypatch.setenv("TILELANG_POKEDEX", "/missing/pokedex")
    with pytest.raises(RuntimeError, match="Set TILELANG_POKEDEX"):
        kernel(a, b)


def test_disk_cache_reconstruction(simulator, tmp_path):
    script = tmp_path / "cache_probe.py"
    script.write_text("""import tilelang, torch
from examples.t1.vector_add import vector_add
from tilelang.t1 import toolchain
import os
if os.environ.get("CACHE_MUST_HIT"):
    def unexpected(*args, **kwargs):
        raise AssertionError("cache hit must not compile")
    toolchain.build = unexpected
kernel = tilelang.compile(vector_add(131), target={"kind": "t1", "codegen": os.environ["T1_TEST_CODEGEN"]}, out_idx=2)
a, b = torch.randn(131), torch.randn(131)
torch.testing.assert_close(kernel(a, b), a+b, rtol=0, atol=0)
assert ('extern "C" void main' if os.environ["T1_TEST_CODEGEN"] == "c" else 'define dllexport void @main') in kernel.get_kernel_source()
assert 't1_runtime_main' in kernel.get_host_source()
""")
    environment = {**os.environ, "TILELANG_CACHE_DIR": str(tmp_path / "cache")}
    modes = ["c", "llvm"] if llvm_version_major(allow_none=True) is not None else ["c"]
    for cached in (False, True):
        if cached:
            environment["CACHE_MUST_HIT"] = "1"
        # Both artifacts must coexist in one cache, then reload without compiling.
        for codegen in modes:
            environment["T1_TEST_CODEGEN"] = codegen
            result = subprocess.run([sys.executable, str(script)], env=environment, text=True, capture_output=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr


def test_toolchain_missing_compiler(monkeypatch):
    monkeypatch.setenv("TILELANG_T1_CLANG", "/missing/clang++")
    with pytest.raises(RuntimeError, match="requires executable"):
        toolchain.compiler()
