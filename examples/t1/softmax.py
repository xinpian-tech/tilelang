"""Stable FP32 row-wise softmax with explicit LLVM/RVV vector reductions."""

import argparse

import torch

import tilelang
import tilelang.t1.language as T


def softmax(rows=256, cols=256, block=128):
    if rows <= 0 or cols <= 0 or not 1 <= block <= 1024:
        raise ValueError("rows and cols must be positive; block must be between 1 and 1024")
    neg_inf = float("-inf")

    @T.prim_func
    def main(A: T.Tensor((rows, cols), "float32"), B: T.Tensor((rows, cols), "float32")):
        with T.Kernel(4) as bx:
            values = T.alloc_local((block,), "float32")
            row_max = T.alloc_var("float32")
            row_sum = T.alloc_var("float32")
            for row_group in T.serial((rows + 3) // 4):
                row = row_group * 4 + bx
                if 0 <= row and row < rows:
                    row_max = neg_inf
                    row_sum = 0.0
                    for chunk in T.serial((cols + block - 1) // block):
                        if chunk < cols // block:
                            for i in T.vectorized(block):
                                values[i] = A[row, chunk * block + i]
                        else:
                            for i in T.vectorized(block):
                                values[i] = neg_inf
                            for i in T.vectorized(cols % block):
                                values[i] = A[row, chunk * block + i]
                        T.reduce_max(values, row_max, dim=0, clear=False)
                    for chunk in T.serial((cols + block - 1) // block):
                        if chunk < cols // block:
                            for i in T.vectorized(block):
                                values[i] = T.exp(A[row, chunk * block + i] - row_max)
                                B[row, chunk * block + i] = values[i]
                        else:
                            for i in T.vectorized(block):
                                values[i] = 0.0
                            for i in T.vectorized(cols % block):
                                values[i] = T.exp(A[row, chunk * block + i] - row_max)
                                B[row, chunk * block + i] = values[i]
                        T.reduce_sum(values, row_sum, dim=0, clear=False)
                    for col in T.Parallel(cols):
                        B[row, col] = B[row, col] / row_sum

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--execution-backend", choices=("pokedex", "xdma"), default="pokedex")
    parser.add_argument("--vlen", type=int)
    args = parser.parse_args()
    vlen = args.vlen or (1024 if args.execution_backend == "xdma" else 256)
    kernel = tilelang.compile(
        softmax(args.rows, args.cols, args.block),
        target={"kind": "t1", "codegen": "llvm", "vlen": vlen},
        execution_backend=args.execution_backend,
        out_idx=-1,
        compile_flags=["-fno-vectorize", "-fno-slp-vectorize"],
    )
    for seed in (7, 19):
        x = torch.randn((args.rows, args.cols), generator=torch.Generator().manual_seed(seed)) * 8
        actual = kernel(x)
        torch.testing.assert_close(actual, torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-7)
    print(f"PASS: softmax ({args.rows}, {args.cols}), four harts, VLEN={vlen}")
    print(f"Device ELF: {kernel.adapter.libpath}")


if __name__ == "__main__":
    main()
