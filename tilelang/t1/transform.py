"""T1 lowering for local reductions and freestanding FP32 exponential."""

from tvm import DataType, IRModule, tirx


def lower_reductions(mod, target):
    """Lower local FP32 sum/max through standard LLVM vector intrinsics."""

    def visit(node):
        if not isinstance(node, tirx.Evaluate) or not isinstance(node.value, tirx.Call):
            return None
        call = node.value
        if getattr(call.op, "name", "") != "tl.tileop.reduce":
            return None
        if str(target.attrs["codegen"]) != "llvm":
            raise ValueError("T1 reductions require codegen='llvm'")
        op = call.op.get_attr("TLOpBuilder")(call.args, call.annotations)
        kind = call.args[2].value
        if kind not in ("sum", "max"):
            raise ValueError("T1 reductions currently support sum and max")
        if op.src.scope() != "local" or op.dst.scope() not in ("local", "local.var"):
            raise ValueError("T1 reductions require local input and local/local.var output")
        if op.src.dtype != "float32" or op.dst.dtype != "float32":
            raise ValueError("T1 reductions require float32 input and output")
        if op.src.data.same_as(op.dst.data):
            raise ValueError("T1 reduction input and output must not alias")
        if op.batch != 1 or any(str(key) not in ("nan_propagate", "batch") for key in op.annotations):
            raise ValueError("T1 reductions support only batch=1 and the nan_propagate annotation")
        src_ranges, dst_ranges = op.srcRegion.region, op.dstRegion.region
        shape = [r.extent for r in src_ranges]
        if any(not isinstance(n, tirx.IntImm) or n.value <= 0 for n in shape):
            raise ValueError("T1 reductions require positive static region extents")
        dim = int(op.dim)
        outer = [tirx.Var(f"reduce_i{i}", "int32") for i in range(len(shape)) if i != dim]
        src_indices = [r.min for r in src_ranges]
        for i, var in zip((i for i in range(len(shape)) if i != dim), outer):
            src_indices[i] += var
        dst_offsets = list(outer)
        if len(dst_ranges) == len(src_ranges):
            dst_offsets.insert(dim, tirx.const(0, "int32"))
        dst_indices = [r.min + offset for r, offset in zip(dst_ranges, dst_offsets)]
        accumulator = tirx.BufferLoad(op.dst, dst_indices)
        scalar_max = "llvm.maximum" if op.nan_propagate else "llvm.maxnum"
        vector_max = "llvm.vector.reduce.fmaximum" if op.nan_propagate else "llvm.vector.reduce.fmax"

        def update(offset, lanes):
            indices = list(src_indices)
            indices[dim] += offset if lanes == 1 else tirx.Ramp(offset, 1, lanes)
            value = tirx.BufferLoad(op.src, indices)
            if kind == "sum":
                result = accumulator + value if lanes == 1 else tirx.call_llvm_pure_intrin(
                    "float32", "llvm.vector.reduce.fadd", accumulator, value
                )
            else:
                reduced = value if lanes == 1 else tirx.call_llvm_pure_intrin("float32", vector_max, value)
                result = tirx.call_llvm_pure_intrin("float32", scalar_max, accumulator, reduced)
            return tirx.BufferStore(op.dst, result, dst_indices)

        count = int(shape[dim])
        lanes = min(int(target.attrs["vlen"]) // 32, count)
        chunk = tirx.Var("reduce_chunk", "int32")
        stmts = []
        identity = tirx.const(0.0 if kind == "sum" else float("-inf"), "float32")
        if op.clear:
            stmts.append(tirx.BufferStore(op.dst, identity, dst_indices))
        if lanes == 1:
            stmts.append(update(0, 1))
        else:
            # Accumulate lane-wise, then reduce horizontally once per tile.
            # This avoids a serial scalar dependency across every input value.
            partial = tirx.decl_buffer((lanes,), "float32", name="reduce_lanes", scope="local")
            vector_indices = [tirx.Ramp(0, 1, lanes)]
            previous = tirx.BufferLoad(partial, vector_indices)
            indices = list(src_indices)
            indices[dim] += tirx.Ramp(chunk * lanes, 1, lanes)
            value = tirx.BufferLoad(op.src, indices)
            combined = previous + value if kind == "sum" else tirx.call_llvm_pure_intrin(value.dtype, scalar_max, previous, value)
            stmts.extend(
                [
                    tirx.AllocBuffer(partial),
                    tirx.BufferStore(partial, tirx.Broadcast(identity, lanes), vector_indices),
                    tirx.For(chunk, 0, count // lanes, tirx.ForKind.SERIAL, tirx.BufferStore(partial, combined, vector_indices)),
                ]
            )
            if kind == "sum":
                result = tirx.call_llvm_pure_intrin("float32", "llvm.vector.reduce.fadd", accumulator, previous)
            else:
                reduced = tirx.call_llvm_pure_intrin("float32", vector_max, previous)
                result = tirx.call_llvm_pure_intrin("float32", scalar_max, accumulator, reduced)
            stmts.append(tirx.BufferStore(op.dst, result, dst_indices))
        if count % lanes:
            stmts.append(update((count // lanes) * lanes, count % lanes))
        body = stmts[0] if len(stmts) == 1 else tirx.SeqStmt(stmts)
        for i, var in reversed(list(zip((i for i in range(len(shape)) if i != dim), outer))):
            body = tirx.For(var, 0, shape[i], tirx.ForKind.SERIAL, body)
        return body

    return IRModule({gv: func.with_body(tirx.stmt_functor.ir_transform(func.body, visit, None)) for gv, func in mod.functions.items()})


def lower_exp(mod):
    """Expand FP32 exp after vectorization; no external math calls are needed.

    x = n * log(2) + r, |r| <= log(2)/2. A degree-seven Taylor
    polynomial approximates exp(r). Splitting 2**n into two normal factors
    covers subnormal results and the upper FP32 exponent without invalid
    exponent bits. Clamp/sanitize before float-to-int conversion, including
    inactive Select arms, so infinities and NaNs cannot create LLVM poison.
    """

    def visit(node):
        if not isinstance(node, tirx.Call) or getattr(node.op, "name", "") != "tirx.exp":
            return None
        dtype = DataType(node.dtype)
        if dtype.type_code != DataType("float32").type_code or dtype.bits != 32:
            raise ValueError("T1 freestanding exp supports float32")
        lanes = dtype.lanes
        int_dtype = "int32" if lanes == 1 else f"int32x{lanes}"
        bindings = []

        def bind(name, value):
            var = tirx.Var(name, value.dtype)
            bindings.append((var, value))
            return var

        def constant(value, scalar_dtype="float32"):
            value = tirx.const(value, scalar_dtype)
            return value if lanes == 1 else tirx.Broadcast(value, lanes)

        x = bind("exp_x", node.args[0])
        isnan = bind("exp_nan", tirx.isnan(x))
        safe = bind("exp_safe", tirx.Select(isnan, constant(0), tirx.min(tirx.max(x, constant(-104)), constant(88.72283172607422))))
        scaled = bind("exp_scaled", safe * constant(1.4426950408889634))
        n = bind("exp_n", tirx.Cast(int_dtype, scaled + tirx.Select(scaled >= constant(0), constant(0.5), constant(-0.5))))
        nf = bind("exp_nf", tirx.Cast(node.dtype, n))
        r = bind("exp_r", (safe - nf * constant(0.693145751953125)) - nf * constant(1.428606765330187e-6))
        polynomial = constant(1 / 5040)
        for coefficient in (1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0):
            polynomial = bind("exp_poly", polynomial * r + constant(coefficient))
        exponent = bind("exp_exponent", tirx.min(tirx.max(n, constant(-126, "int32")), constant(127, "int32")))

        def power_of_two(value):
            bits = tirx.shift_left(value + constant(127, "int32"), constant(23, "int32"))
            return tirx.reinterpret(node.dtype, bits)

        result = (polynomial * power_of_two(exponent)) * power_of_two(n - exponent)
        result = tirx.Select(x < constant(-104), constant(0), result)
        result = tirx.Select(x > constant(88.72283172607422), constant(float("inf")), result)
        result = tirx.Select(isnan, x, result)
        for var, value in reversed(bindings):
            result = tirx.Let(var, value, result)
        return result

    return IRModule({gv: func.with_body(tirx.stmt_functor.ir_transform(func.body, None, visit)) for gv, func in mod.functions.items()})
