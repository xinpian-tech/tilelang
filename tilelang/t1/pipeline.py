"""T1's initial global-buffer, scalar-loop lowering pipeline.

One frontend function describes one four-block launch. The device entry takes
the original buffers followed by the block ID; no host function is generated.
"""

from tvm import IRModule, tirx
from tvm.ir import CallingConv

import tilelang


@tirx.transform.prim_func_pass(opt_level=0)
def prepare(func, mod, ctx):
    if len(func.params) != len(func.buffer_map):
        raise ValueError("T1 currently accepts only static float32 tensor arguments, not scalar arguments")
    for buffer in func.buffer_map.values():
        if buffer.dtype != "float32" or buffer.scope() != "global":
            raise ValueError("T1 currently supports only global float32 buffers")
        if any(not isinstance(dim, tirx.IntImm) or dim.value <= 0 for dim in buffer.shape):
            raise ValueError("T1 requires positive static tensor shapes")
        strides = []
        stride = 1
        for dim in reversed(buffer.shape):
            strides.insert(0, stride)
            stride *= int(dim)
        if (
            (
                buffer.strides
                and (
                    len(buffer.strides) != len(strides)
                    or any(not isinstance(s, tirx.IntImm) or int(s) != e for s, e in zip(buffer.strides, strides))
                )
            )
            or not isinstance(buffer.elem_offset, tirx.IntImm)
            or buffer.elem_offset.value != 0
        ):
            raise ValueError("T1 requires contiguous buffers with zero element offset")

    launch = func.body
    while isinstance(launch, tirx.SBlockRealize):
        launch = launch.block.body
    if not isinstance(launch, tirx.AttrStmt) or launch.attr_key != "thread_extent":
        raise ValueError("T1 requires the entire function body to be one T.Kernel(4) launch")

    grid = []

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            if node.node.thread_tag != "blockIdx.x" or not isinstance(node.value, tirx.IntImm) or node.value.value != 4:
                raise ValueError("T1 requires exactly one one-dimensional T.Kernel(4) launch")
            grid.append(node.node.var)
            return node.body
        if isinstance(node, tirx.For):
            if node.kind == tirx.ForKind.PARALLEL:
                return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body)
            if node.kind not in (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED):
                raise ValueError("T1 supports serial/parallel elementwise loops; explicit vector and thread loops are unsupported")
        if isinstance(node, tirx.PrimExpr):
            dtype = str(node.dtype)
            if dtype not in ("handle", "bool", "int32", "uint32", "float32"):
                raise ValueError(f"T1 does not yet support scalar expression dtype {dtype}")
        if isinstance(node, (tirx.Min, tirx.Max, tirx.FloorDiv, tirx.FloorMod)):
            raise ValueError("T1 min/max and floor division/modulo expressions are not implemented yet")
        if isinstance(node, tirx.Mod) and str(node.dtype) == "float32":
            raise ValueError("T1 floating-point modulo is not implemented yet")
        if isinstance(node, tirx.SBlock) and (node.init is not None or node.iter_vars or node.match_buffers):
            raise ValueError("T1 reduction blocks and buffer aliases are not implemented yet")
        if isinstance(node, tirx.Call):
            name = getattr(node.op, "name", str(node.op))
            if name not in ("tirx.if_then_else", "tirx.likely"):
                raise ValueError(f"T1 does not yet support operation {name}; use global-buffer elementwise expressions")
        if isinstance(node, tirx.AllocBuffer) or (isinstance(node, tirx.SBlock) and node.alloc_buffers):
            raise ValueError("T1 local/shared allocations are not implemented yet; use global buffers")
        return None

    body = tirx.stmt_functor.ir_transform(func.body, None, visit)
    if len(grid) != 1:
        raise ValueError("T1 requires exactly one one-dimensional T.Kernel(4) launch")
    # Device buffer pointers replace the original frontend handle parameters.
    # The body already refers to these Buffer.data variables.
    params = [func.buffer_map[param].data for param in func.params] + grid
    result = tirx.PrimFunc(params, body, func.ret_type, {}, func.attrs)
    return result.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH).with_attr("t1.grid", 4)


def lower(mod: IRModule, target):
    if len(mod.functions) != 1:
        raise ValueError("T1 currently supports one function per compilation")
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        unsupported_annotations=["cluster_dims", "tl.launch_threads", "pragma_import_c"],
    )(mod)
    mod = prepare(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    return mod
