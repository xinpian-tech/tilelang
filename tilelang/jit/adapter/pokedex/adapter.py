"""Synchronous ELF subprocess execution with explicit simulated-memory copies."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tilelang.jit.adapter.t1 import T1KernelAdapter
from tilelang.t1 import toolchain


class PokedexKernelAdapter(T1KernelAdapter):
    def _execute(self, memory):
        executable = shutil.which(os.environ.get("TILELANG_POKEDEX", "pokedex"))
        model = os.environ.get("POKEDEX_MODEL_LIB")
        if executable is None or not model or not Path(model).is_file():
            raise RuntimeError("Set TILELANG_POKEDEX to a run-gpgpu executable and POKEDEX_MODEL_LIB to its model library")
        budget = int(os.environ.get("TILELANG_POKEDEX_MAX_INSTRUCTIONS", "10000000"))
        if budget <= 0:
            raise ValueError("TILELANG_POKEDEX_MAX_INSTRUCTIONS must be positive")
        with tempfile.TemporaryDirectory(prefix="tilelang-pokedex-call-") as directory:
            directory = Path(directory)
            source = directory / "input.bin"
            output = directory / "output.bin"
            source.write_bytes(memory)
            command = [
                executable,
                "run-gpgpu",
                self.libpath,
                "--model-lib",
                model,
                "--vlen",
                str(int(self.target.attrs["vlen"])),
                "--max-instructions",
                str(budget),
                "--load-memory",
                f"{toolchain.DATA_BASE:#x}:{source}",
                "--dump-memory",
                f"{toolchain.DATA_BASE:#x}:{self.total_size}:{output}",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Pokedex exceeded the 60-second wall-time limit") from error
            if result.returncode:
                raise RuntimeError(f"Pokedex kernel failed (exit {result.returncode}):\n{result.stderr}")
            if not output.is_file() or output.stat().st_size != self.total_size:
                raise RuntimeError("Pokedex returned missing or truncated output memory")
            memory = bytearray(output.read_bytes())
        return memory
