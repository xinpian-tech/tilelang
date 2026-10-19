"""Compile a TileLang tile to RVV and execute four tiles in Pokedex run-gpgpu.

This is a standalone prototype using the CPU lowering pipeline, not a registered
GPGPU backend or JIT adapter. Each hart calls the generated kernel on one tile.
"""

import argparse
from pathlib import Path
import re
import subprocess

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm


def make_vector_add(block_size):
    @T.prim_func
    def vector_add(
        A: T.Tensor((block_size,), "float32"),
        B: T.Tensor((block_size,), "float32"),
        C: T.Tensor((block_size,), "float32"),
    ):
        for i in T.Parallel(block_size):
            C[i] = A[i] + B[i]

    return vector_add


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pokedex", required=True, help="Pokedex executable with run-gpgpu")
    parser.add_argument("--model-lib", required=True, type=Path)
    parser.add_argument("--clang", default="clang++")
    parser.add_argument("--objdump", default="llvm-objdump")
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=257, help="Elements per hart (four harts)")
    parser.add_argument("--build-dir", type=Path, default=Path("build/gpgpu-demo"))
    args = parser.parse_args()
    if args.vlen < 128 or args.vlen > 65536 or args.vlen & (args.vlen - 1):
        parser.error("--vlen must be a power of two between 128 and 65536")
    if not 64 <= args.block_size <= 65536:
        parser.error("--block-size must be between 64 and 65536")

    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    sources = Path(__file__).resolve().parent
    # Preserve scalar loops in the portable C++ output. LLVM, rather than the
    # CPU backend's fixed-width vec_type helpers, performs RVV vectorization.
    with tvm.target.Target("c"), tvm.transform.PassContext(config={"tirx.disable_vectorize": True}):
        artifact = tilelang.lower(make_vector_add(args.block_size), target="c", target_host="c")
    source = artifact.kernel_source
    (build / "kernel.original.cpp").write_text(source)
    (build / "kernel.tir").write_text(artifact.device_mod.script())

    # This scalar float32/int32 kernel uses no CPU runtime helpers. Replace its
    # sole hosted-runtime include so a freestanding cross build needs no C++ SDK.
    # Keep the unmodified output above for inspection; this is demo-only glue.
    header = "#include <tl_templates/cpp/common.h>"
    if source.count(header) != 1:
        raise RuntimeError("Unexpected CPU codegen preamble; inspect kernel.original.cpp")
    source = source.replace(header, "#include <stdint.h>")
    (build / "kernel.cpp").write_text(source)

    flags = [
        args.clang,
        "--target=riscv32-unknown-elf",
        f"-march=rv32imafc_zicsr_zve32f_zvl{args.vlen}b",
        "-mabi=ilp32f",
        "-mcmodel=medany",
        "-mno-relax",
        "-ffreestanding",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-stack-protector",
        "-O3",
    ]
    run([*flags, "-c", build / "kernel.cpp", "-o", build / "kernel.o"])
    # A scalar verifier is independent of the kernel's vector execution path.
    run(
        [
            *flags,
            "-fno-vectorize",
            "-fno-slp-vectorize",
            f"-DBLOCK_SIZE={args.block_size}",
            "-c",
            sources / "guest.cpp",
            "-o",
            build / "guest.o",
        ]
    )
    run([*flags, "-c", sources / "start.S", "-o", build / "start.o"])
    elf = build / "vector_add.elf"
    run(
        [
            *flags,
            "-nostdlib",
            "-fuse-ld=lld",
            f"-Wl,-T,{sources / 'link.ld'}",
            build / "start.o",
            build / "guest.o",
            build / "kernel.o",
            "-o",
            elf,
        ]
    )
    # Inspect the kernel object, not the harness, to establish that LLVM emitted
    # RVV arithmetic and memory operations for the TileLang-generated function.
    assembly = subprocess.check_output([args.objdump, "-d", str(build / "kernel.o")], text=True)
    (build / "kernel.asm").write_text(assembly)
    for instruction in (r"\bvseti?vli\b", r"\bvle32\.v\b", r"\bvfadd\.vv\b", r"\bvse32\.v\b"):
        if not re.search(instruction, assembly):
            raise RuntimeError(f"Expected RVV instruction {instruction!r} missing; inspect kernel.asm")

    run(
        [
            args.pokedex,
            "run-gpgpu",
            elf,
            "--model-lib",
            args.model_lib.resolve(),
            "--vlen",
            str(args.vlen),
            "--max-instructions",
            "10000000",
            "--output",
            build / "trace.txt",
        ]
    )
    print(f"PASS: 4 harts × {args.block_size} float32 elements; RVV verified; artifacts in {build}")


if __name__ == "__main__":
    main()
