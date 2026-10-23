"""Explicit vector reductions, freestanding exp, and FPGA/simulator softmax."""

import os
import re
import subprocess

import pytest
import torch
import tilelang
import tilelang.t1.language as T
from examples.t1.softmax import softmax
from tvm.target.codegen import llvm_version_major


pytestmark = pytest.mark.skipif(llvm_version_major(allow_none=True) is None, reason="LLVM disabled")


@pytest.fixture(params=["pokedex", "xdma"])
def runtime(request, monkeypatch):
    if request.param == "pokedex":
        if not os.environ.get("TILELANG_POKEDEX") or not os.environ.get("POKEDEX_MODEL_LIB"):
            pytest.skip("set TILELANG_POKEDEX and POKEDEX_MODEL_LIB for simulator tests")
        return "pokedex", 256
    if os.environ.get("TILELANG_TEST_XDMA") != "1":
        pytest.skip("set TILELANG_TEST_XDMA=1 to reset/run the t1_demo FPGA")
    monkeypatch.setenv("TILELANG_XDMA_DRAIN_DELAY", "0")
    return "xdma", 1024


def compile_kernel(func, runtime):
    backend, vlen = runtime
    return tilelang.compile(
        func,
        target={"kind": "t1", "codegen": "llvm", "vlen": vlen},
        execution_backend=backend,
        out_idx=-1,
        compile_flags=["-fno-vectorize", "-fno-slp-vectorize"],
    )


def exp_kernel(n):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.Parallel((n + 3) // 4):
                index = bx * ((n + 3) // 4) + i
                if index < n:
                    B[index] = T.exp(A[index])

    return main


def test_softmax_llvm_source():
    artifact = tilelang.lower(softmax(), target={"kind": "t1", "codegen": "llvm", "vlen": 256})
    source = artifact.kernel_source
    assert "llvm.vector.reduce.fadd.v8f32" in source
    assert "llvm.vector.reduce.fmax.v8f32" in source
    assert "llvm.exp" not in source
    assert "expf" not in source


def test_exp_accuracy_and_special_values(runtime):
    boundaries = torch.tensor(
        [-float("inf"), -105, -104, -103.97208, -103.9, -87.33655, -0.0, 0, 88.72283172607422, 88.72283935546875, 100, float("inf"), float("nan")]
    )
    a = torch.cat([torch.linspace(-104, 88.72, 4096), boundaries])
    kernel = compile_kernel(exp_kernel(a.numel()), runtime)
    actual = kernel(a)
    expected = torch.exp(a.double()).float()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=torch.finfo(torch.float32).smallest_normal * 2**-23, equal_nan=True)
    assert actual[-2].isposinf()
    assert actual[-1].isnan()
    assert actual[4096] == 0
    dump = subprocess.check_output(["llvm-objdump", "-d", kernel.adapter.libpath], text=True)
    assert re.search(r"\bvfm(add|acc)\.v[vf]\b", dump)
    undefined = subprocess.check_output(["llvm-nm", "--undefined-only", kernel.adapter.libpath], text=True)
    assert not undefined.strip()


@pytest.mark.parametrize(("rows", "cols", "block"), [(1, 1, 8), (3, 7, 16), (5, 33, 32), (7, 259, 64), (4, 1025, 128), (1, 4097, 128)])
def test_softmax_execution(runtime, rows, cols, block):
    kernel = compile_kernel(softmax(rows, cols, block), runtime)
    generator = torch.Generator().manual_seed(17)
    inputs = [
        torch.randn((rows, cols), generator=generator) * 8,
        torch.randn((rows, cols), generator=generator) + 10000,
        torch.full((rows, cols), -10000.0),
    ]
    peak = torch.full((rows, cols), -torch.finfo(torch.float32).max)
    peak[:, cols // 2] = torch.finfo(torch.float32).max
    inputs.append(peak)
    for a in inputs:
        actual = kernel(a)
        torch.testing.assert_close(actual, torch.softmax(a, dim=-1), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(actual.sum(dim=-1), torch.ones(rows), rtol=1e-6, atol=1e-6)
        assert (actual >= 0).all()
    if cols == 259:
        for value in (float("nan"), float("inf"), -float("inf")):
            a = torch.full((rows, cols), value)
            torch.testing.assert_close(kernel(a), torch.softmax(a, dim=-1), equal_nan=True)
    if block >= runtime[1] // 32:
        dump = subprocess.check_output(["llvm-objdump", "-d", kernel.adapter.libpath], text=True)
        assert re.search(r"\bvfredmax\.vs\b", dump)
        assert re.search(r"\bvfredosum\.vs\b", dump)


def reduction_kernel(n, kind, clear):
    @T.prim_func
    def main(A: T.Tensor((4, n), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(4) as bx:
            values = T.alloc_local((n,), "float32")
            result = T.alloc_var("float32", init=3.0)
            for i in T.vectorized(n):
                values[i] = A[bx, i]
            if kind == "sum":
                T.reduce_sum(values, result, clear=clear)
            else:
                T.reduce_max(values, result, clear=clear)
            B[bx] = result

    return main


@pytest.mark.parametrize(("n", "kind", "clear"), [(1, "sum", True), (11, "sum", True), (37, "sum", False), (11, "max", True), (37, "max", False)])
def test_reductions(runtime, n, kind, clear):
    a = torch.arange(4 * n, dtype=torch.float32).reshape(4, n) - 2 * n
    kernel = compile_kernel(reduction_kernel(n, kind, clear), runtime)
    expected = a.sum(dim=1) if kind == "sum" else a.max(dim=1).values
    if not clear:
        expected = expected + 3 if kind == "sum" else torch.maximum(expected, torch.tensor(3.0))
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)
