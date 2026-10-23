"""T1 freestanding ELF toolchain and versioned four-hart launch ABI."""

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

ABI_VERSION = 2
DATA_BASE = 0x90000000
DATA_LIMIT = 0x98000000
RUNTIME = Path(__file__).with_name("runtime")


def required_tool(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"T1 requires executable {name!r}; install it or configure TILELANG_T1_CLANG")
    # LLVM's multicall drivers select their mode from argv[0]. In particular,
    # resolving ld.lld -> lld makes even --version fail on some installations.
    return str(Path(path).absolute())


def compiler():
    return required_tool(os.environ.get("TILELANG_T1_CLANG", "clang++"))


def cache_identity():
    from tvm.target.codegen import llvm_version_major

    clang = compiler()
    linker = required_tool("ld.lld")
    files = sorted(Path(__file__).parent.rglob("*.py")) + sorted(RUNTIME.iterdir())
    files += sorted((Path(__file__).parents[1] / "jit" / "adapter" / "pokedex").glob("*.py"))
    files += sorted((Path(__file__).parents[1] / "jit" / "adapter" / "xdma").glob("*.py"))
    files += [Path(__file__).parents[1] / "jit" / "adapter" / "t1.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return {
        "abi": ABI_VERSION,
        "sources": digest.hexdigest(),
        "llvm_major": llvm_version_major(allow_none=True),
        "compiler": [clang, subprocess.check_output([clang, "--version"], text=True)],
        "linker": [linker, subprocess.check_output([linker, "--version"], text=True)],
    }


def buffer_layout(params):
    offsets, sizes = [], []
    end = 0
    for param in params:
        if str(param.dtype) != "float32" or not param.shape:
            raise ValueError("T1 currently supports only static float32 tensor arguments")
        shape = [int(dim) for dim in param.shape]
        if any(dim <= 0 for dim in shape):
            raise ValueError("T1 tensor dimensions must be positive")
        size = math.prod(shape) * 4
        offset = (end + 63) // 64 * 64
        end = offset + size
        if DATA_BASE + end > DATA_LIMIT:
            raise ValueError("T1 arguments exceed the 128 MiB transfer arena")
        offsets.append(offset)
        sizes.append(size)
    return offsets, sizes, end


def launcher(symbol, params, *, runtime="pokedex", vlen=256):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) or symbol.startswith("t1_runtime_"):
        raise ValueError("T1 kernel symbol must be a C identifier outside the t1_runtime_ namespace")
    offsets, _, _ = buffer_layout(params)
    arguments = ", ".join([f"(float*)0x{DATA_BASE + offset:x}u" for offset in offsets] + ["(int32_t)hart"])
    signature = ", ".join(["float*"] * len(params) + ["int32_t"])
    if runtime == "fpga":
        return f"""#include <stdint.h>
#include "fpga_protocol.h"
extern "C" void {symbol}({signature});
extern "C" void t1_runtime_main(uint32_t hart) {{
  volatile uint32_t* record = (volatile uint32_t*)(T1_MAILBOX + hart * T1_RECORD_SIZE);
  uint32_t vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  record[T1_MB_ABI / 4] = T1_FPGA_ABI;
  record[T1_MB_HART / 4] = hart;
  record[T1_MB_VLENB / 4] = vlenb;
  if (vlenb != {vlen // 8}u) {{
    asm volatile("fence iorw, iorw" ::: "memory");
    record[T1_MB_STATUS / 4] = T1_STATUS_CONFIG;
  }} else {{
    {symbol}({arguments});
    asm volatile("fence iorw, iorw" ::: "memory");
    record[T1_MB_STATUS / 4] = T1_STATUS_PASS;
  }}
  asm volatile("fence iorw, iorw" ::: "memory");
  for (;;) asm volatile("nop");
}}
"""
    if runtime != "pokedex":
        raise ValueError(f"Unknown T1 runtime: {runtime}")
    return f"""#include <stdint.h>
extern "C" void {symbol}({signature});
extern "C" volatile uint32_t tohost;
static volatile uint32_t t1_runtime_done[4];
extern "C" void t1_runtime_main(uint32_t hart) {{
  {symbol}({arguments});
  asm volatile("fence rw, w" ::: "memory");
  t1_runtime_done[hart] = 1;
  if (hart == 0) {{
    for (uint32_t h = 0; h < 4; ++h)
      while (!t1_runtime_done[h]) asm volatile("nop");
    asm volatile("fence rw, rw" ::: "memory");
    tohost = 1;
  }}
  for (;;) asm volatile("nop");
}}
"""


def build(source, wrapper, target, directory, compile_flags=None, *, runtime="pokedex"):
    if runtime not in ("pokedex", "fpga"):
        raise ValueError(f"Unknown T1 runtime: {runtime}")
    clang = compiler()
    required_tool("ld.lld")
    flags = [
        clang,
        "--target=riscv32-unknown-elf",
        f"-march=rv32imafc_zicsr_zve32f_zvl{int(target.attrs['vlen'])}b",
        "-mabi=ilp32f",
        "-mcmodel=medany",
        "-mno-relax",
        "-ffreestanding",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-stack-protector",
        "-O3",
        f"-I{RUNTIME}",
    ]
    directory = Path(directory)
    kernel_file = directory / ("kernel.ll" if str(target.attrs["codegen"]) == "llvm" else "kernel.cpp")
    kernel_file.write_text(source)
    (directory / "launcher.cpp").write_text(wrapper)

    def invoke(args):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"T1 compiler failed:\n{result.stderr}")

    extra = list(compile_flags or [])
    invoke([*flags, *extra, "-c", str(kernel_file), "-o", str(directory / "kernel.o")])
    invoke([*flags, "-c", str(directory / "launcher.cpp"), "-o", str(directory / "launcher.o")])
    startup = "fpga_start.S" if runtime == "fpga" else "start.S"
    linker_script = "fpga_link.ld" if runtime == "fpga" else "link.ld"
    invoke([*flags, "-c", str(RUNTIME / startup), "-o", str(directory / "start.o")])
    output = directory / "kernel.elf"
    invoke(
        [
            *flags,
            "-nostdlib",
            "-fuse-ld=lld",
            f"-Wl,-T,{RUNTIME / linker_script}",
            str(directory / "start.o"),
            str(directory / "launcher.o"),
            str(directory / "kernel.o"),
            "-o",
            str(output),
        ]
    )
    return str(output)
