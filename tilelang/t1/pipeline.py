"""T1's four-hart lowering, with C and explicit LLVM vector paths.

One frontend function describes one four-block launch. The device entry takes
the original buffers followed by the block ID; no host function is generated.
"""

from tvm import DataType, IRModule, tirx, transform
from tvm.ir import CallingConv

import tilelang
from .target import llvm_target


@tirx.transform.prim_func_pass(opt_level=0)
def prepare(func, mod, ctx):
    explicit = str(func.attrs["target"].attrs["codegen"]) == "llvm"
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
            if node.kind == tirx.ForKind.PARALLEL and not explicit:
                return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body)
            allowed = (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED)
            if explicit:
                allowed += (tirx.ForKind.PARALLEL, tirx.ForKind.VECTORIZED)
            if node.kind not in allowed:
                raise ValueError("T1 supports serial/parallel elementwise loops; explicit vector and thread loops are unsupported")
        if isinstance(node, tirx.SBlock) and (node.init is not None or node.iter_vars or node.match_buffers):
            raise ValueError("T1 reduction blocks and buffer aliases are not implemented yet")
        if isinstance(node, tirx.Call):
            name = getattr(node.op, "name", str(node.op))
            lowering_targets = ("llvm" if explicit else "t1", "default")
            if name.startswith("tl.") and not any(
                node.op.has_attr(f"{target}.{rule}") and node.op.get_attr(f"{target}.{rule}") is not None
                for target in lowering_targets
                for rule in ("FLowerIntrinsic", "FLegalize")
            ):
                raise ValueError(f"T1 does not yet support operation {name}; TileLang operation lowering is required")
        buffers = []
        if isinstance(node, tirx.AllocBuffer):
            buffers = [node.buffer]
        elif isinstance(node, tirx.SBlock):
            buffers = node.alloc_buffers
        for buffer in buffers:
            if buffer.scope() not in ("local", "local.var"):
                raise ValueError(f"T1 allocations in scope {buffer.scope()!r} are not implemented; use local storage")
        return None

    body = tirx.stmt_functor.ir_transform(func.body, None, visit)
    if len(grid) != 1:
        raise ValueError("T1 requires exactly one one-dimensional T.Kernel(4) launch")
    # Device buffer pointers replace the original frontend handle parameters.
    # The body already refers to these Buffer.data variables.
    params = [func.buffer_map[param].data for param in func.params] + grid
    result = tirx.PrimFunc(params, body, func.ret_type, {}, func.attrs)
    return result.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH).with_attr("t1.grid", 4)


def vectorize(mod, target):
    """Strip-mine innermost parallel loops and materialize vector TIR operations."""
    internal_target = llvm_target(target)
    width = int(target.attrs["vlen"]) // 32

    def split(node):
        if not isinstance(node, tirx.For) or node.kind not in (tirx.ForKind.PARALLEL, tirx.ForKind.VECTORIZED):
            return None
        nested = []
        tirx.stmt_functor.post_order_visit(node.body, lambda child: nested.append(child) if isinstance(child, tirx.For) else None)
        if nested:
            if node.kind == tirx.ForKind.VECTORIZED:
                raise ValueError("T1 explicit vector loops must be innermost")
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body)
        if not isinstance(node.extent, tirx.IntImm):
            raise ValueError("T1 LLVM vectorization requires constant loop extents")
        extent = int(node.extent)
        lanes = min(width, extent)
        if lanes <= 1:
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body)
        chunk = tirx.Var(f"{node.loop_var.name}_chunk", "int32")
        lane = tirx.Var(f"{node.loop_var.name}_lane", "int32")
        index = node.min + chunk * lanes + lane
        body = tirx.stmt_functor.substitute(node.body, {node.loop_var: index})
        body = tirx.For(lane, 0, lanes, tirx.ForKind.VECTORIZED, body)
        body = tirx.AttrStmt(lane, "t1.vectorized_region", 1, body)
        body = tirx.For(chunk, 0, extent // lanes, tirx.ForKind.SERIAL, body)
        remainder = extent % lanes
        if remainder:
            tail_index = node.min + (extent // lanes) * lanes + lane
            tail = tirx.stmt_functor.substitute(node.body, {node.loop_var: tail_index})
            tail = tirx.For(lane, 0, remainder, tirx.ForKind.SERIAL, tail)
            body = tirx.SeqStmt([body, tail])
        return body

    functions = {}
    for gv, func in mod.functions.items():
        body = tirx.stmt_functor.ir_transform(func.body, None, split)
        functions[gv] = func.with_body(body).with_attr("target", internal_target)
    mod = IRModule(functions)
    ctx = transform.PassContext.current()
    with transform.PassContext(opt_level=ctx.opt_level, config={**ctx.config, "tirx.enable_buffer_level_predication": True}):
        mod = tirx.transform.VectorizeLoop()(mod)

    # TVM may scalarize unsupported vector control flow. Make this observable.
    def verify(node):
        if not isinstance(node, tirx.AttrStmt) or node.attr_key != "t1.vectorized_region":
            return None
        stores = []
        scalarized = []
        tirx.stmt_functor.post_order_visit(
            node.body,
            lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) and DataType(node.value.dtype).lanes > 1 else None,
        )
        tirx.stmt_functor.post_order_visit(node.body, lambda child: scalarized.append(child) if isinstance(child, tirx.For) else None)
        if not stores or scalarized:
            raise ValueError("T1 LLVM could not explicitly vectorize a selected loop; simplify its control flow or use codegen='c'")
        return node.body

    return IRModule(
        {
            gv: func.with_body(tirx.stmt_functor.ir_transform(func.body, None, verify)).with_attr("target", target)
            for gv, func in mod.functions.items()
        }
    )


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
    if str(target.attrs["codegen"]) == "llvm":
        mod = vectorize(mod, target)
        # Select LLVM intrinsic rules before the common codegen preparation can
        # lower math operations to default external library calls under `t1`.
        internal_target = llvm_target(target)
        mod = IRModule({gv: func.with_attr("target", internal_target) for gv, func in mod.functions.items()})
        mod = tilelang.transform.LowerIntrin()(mod)
        mod = IRModule({gv: func.with_attr("target", target) for gv, func in mod.functions.items()})
    return mod
