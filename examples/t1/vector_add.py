"""Four-hart RVV vector addition through the native T1 backend."""

import argparse
import torch
import tilelang
import tilelang.t1.language as T


def vector_add(n=1025):
    block = (n + 3) // 4

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.Parallel(block):
                index = bx * block + i
                if index < n:
                    C[index] = A[index] + B[index]

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1025)
    parser.add_argument("--vlen", type=int, default=256)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    kernel = tilelang.compile(vector_add(args.size), target={"kind": "t1", "vlen": args.vlen}, execution_backend="pokedex", out_idx=-1)
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        a = torch.randn(args.size, generator=generator)
        b = torch.randn(args.size, generator=generator)
        actual = kernel(a, b)
        torch.testing.assert_close(actual, a + b, rtol=0, atol=0)
    print(f"PASS: {args.size} float32 elements, four harts, VLEN={args.vlen}, two different input pairs")
    print(f"Device ELF: {kernel.adapter.libpath}")


if __name__ == "__main__":
    main()
