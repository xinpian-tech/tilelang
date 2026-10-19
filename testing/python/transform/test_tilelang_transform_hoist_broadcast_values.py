import tilelang
import tilelang.language as T
import torch
import re
import pytest
import tilelang.testing
from tilelang import tvm as tvm
import tilelang as tl


@tilelang.jit
def qwq(dtype=torch.float8_e4m3fn):
    @T.prim_func
    def main(
        A: T.Tensor((32,), dtype),
        B: T.Tensor((16,), dtype),
        C: T.Tensor((8,), dtype),
        D: T.Tensor((4,), dtype),
        E: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, threads=32):
            var = T.alloc_var(dtype, 1.0)
            for i in T.vectorized(32):
                A[i] = var
            for i in T.vectorized(16):
                B[i] = 13.5
            for i in T.vectorized(8):
                C[i] = 3.14
            for i in T.vectorized(4):
                D[i] = 2.72
            for i in T.vectorized(2):
                E[i] = 430

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu, torch.float16])
def test_hoist_broadcast(dtype):
    kernel = qwq(dtype)
    print(kernel.get_kernel_source())
    matches = re.findall(r"(\w+) broadcast_var(_[0-9]+)? = \1", kernel.get_kernel_source())
    assert len(matches) == 4
    a = torch.empty((32,), device="cuda", dtype=dtype)
    b = torch.empty((16,), device="cuda", dtype=dtype)
    c = torch.empty((8,), device="cuda", dtype=dtype)
    d = torch.empty((4,), device="cuda", dtype=dtype)
    e = torch.empty((2,), device="cuda", dtype=dtype)
    kernel(a, b, c, d, e)


def _check(original, transformed):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.HoistBroadcastValues()(mod)

    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tirx.transform.BindTarget(target)(transformed)

    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_transform_hoist():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            A_shared[0:8] = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            broadcast_var: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_var_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            A_shared[0:8] = T.Broadcast(broadcast_var, 8) + T.Broadcast(broadcast_var_1, 8)

    _check(before, after)


def test_hoist_preserves_store_predicate():
    tir = tvm.tirx
    buffer = tir.decl_buffer((8,), "float32")
    mask = tir.Var("mask", "uint1x8")
    store = tir.BufferStore(buffer, tir.Broadcast(tir.const(1.0, "float32"), 8), [tir.Ramp(0, 1, 8)], predicate=mask)
    func = tir.PrimFunc([buffer.data, mask], store)
    mod = tl.transform.HoistBroadcastValues()(tvm.IRModule({"main": func}))
    stores = []
    tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: stores.append(node) if isinstance(node, tir.BufferStore) else None)
    assert len(stores) == 1
    tvm.ir.assert_structural_equal(stores[0].predicate, mask)


def test_transform_hoist_let_stmt():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            val: T.float8_e4m3fnx8 = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)
            A_shared[0:8] = val

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            broadcast_var: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_var_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            val: T.float8_e4m3fnx8 = T.Broadcast(broadcast_var, 8) + T.Broadcast(broadcast_var_1, 8)
            A_shared[0:8] = val

    _check(before, after)


def test_transform_hoist_let_stmt_with_nested_bufferstore_broadcasts():
    """Test case for the bug where BufferStore in LetStmt body clears pending_defs.

    This test validates that broadcasts hoisted from a LetStmt's value expression
    are preserved even when the body contains a BufferStore with additional broadcasts.
    """

    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            # LetStmt value has broadcasts
            val: T.float8_e4m3fnx8 = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)
            # Body is a BufferStore with additional broadcasts
            A_shared[0:8] = val + T.Broadcast(T.float8_e4m3fn(5.6), 8)

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            # Hoisted from LetStmt value
            broadcast_var: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_var_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            val: T.float8_e4m3fnx8 = T.Broadcast(broadcast_var, 8) + T.Broadcast(broadcast_var_1, 8)
            # Hoisted from BufferStore
            broadcast_var_2: T.float8_e4m3fn = T.float8_e4m3fn(5.6)
            A_shared[0:8] = val + T.Broadcast(broadcast_var_2, 8)

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
