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

    with pytest.raises(ValueError, match="allocations are not implemented"):
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


@pytest.mark.parametrize(("n", "vlen"), [(1025, 256), (259, 128), (3, 256)])
def test_execution_repeated_inputs_and_tail(simulator, n, vlen):
    kernel = tilelang.compile(vector_add(n), target={"kind": "t1", "vlen": vlen}, out_idx=-1)
    for seed in (7, 19):
        generator = torch.Generator().manual_seed(seed)
        a, b = (torch.randn(n, generator=generator) for _ in range(2))
        torch.testing.assert_close(kernel(a, b), a + b, rtol=0, atol=0)
    if n == 1025:
        dump = subprocess.check_output(["llvm-objdump", "-d", kernel.adapter.libpath], text=True)
        assert re.search(r"\bvle32\.v\b", dump)
        assert re.search(r"\bvfadd\.vv\b", dump)
        assert re.search(r"\bvse32\.v\b", dump)


def test_four_block_ids_and_explicit_outputs(simulator):
    @T.prim_func
    def main(A: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            A[bx] = T.Cast("float32", bx)

    kernel = tilelang.compile(main, target="t1")
    output = torch.full((4,), -1.0)
    assert kernel(output) is None
    torch.testing.assert_close(output, torch.arange(4, dtype=torch.float32))


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
kernel = tilelang.compile(vector_add(131), target="t1", out_idx=2)
a, b = torch.randn(131), torch.randn(131)
torch.testing.assert_close(kernel(a, b), a+b, rtol=0, atol=0)
assert 'extern "C" void main' in kernel.get_kernel_source()
assert 't1_runtime_main' in kernel.get_host_source()
""")
    environment = {**os.environ, "TILELANG_CACHE_DIR": str(tmp_path / "cache")}
    for cached in (False, True):
        if cached:
            environment["CACHE_MUST_HIT"] = "1"
        result = subprocess.run([sys.executable, str(script)], env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


def test_toolchain_missing_compiler(monkeypatch):
    monkeypatch.setenv("TILELANG_T1_CLANG", "/missing/clang++")
    with pytest.raises(RuntimeError, match="requires executable"):
        toolchain.compiler()
