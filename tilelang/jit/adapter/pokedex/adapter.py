"""Synchronous ELF subprocess execution with explicit simulated-memory copies."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.t1 import toolchain
from tilelang.transform import PassConfigKey


class PokedexKernelAdapter(BaseKernelAdapter):
    def __init__(self, params, result_idx, target, func_or_mod, device_kernel_source, pass_configs=None, compile_flags=None):
        self._initialize(params, result_idx, target, func_or_mod)
        self.device_kernel_source = device_kernel_source
        self._build_dir = tempfile.TemporaryDirectory(prefix="tilelang-t1-build-")
        # JIT merges compile_flags into pass_configs before creating the adapter.
        flags = (pass_configs or {}).get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, compile_flags)
        if isinstance(flags, str):
            flags = [flags]
        self.libpath = toolchain.build(self.device_kernel_source, self.host_kernel_source, target, self._build_dir.name, flags)

    def _initialize(self, params, result_idx, target, func):
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        if len(set(self.result_idx)) != len(self.result_idx):
            raise ValueError("Pokedex out_idx must not contain duplicate indices")
        self.target = target
        self.offsets, self.sizes, self.total_size = toolchain.buffer_layout(params)
        self.host_kernel_source = toolchain.launcher(str(func.attrs["global_symbol"]), params)
        self.func = self._convert_torch_func()

    @classmethod
    def from_database(cls, params, result_idx, target, func_or_mod, device_kernel_source, kernel_lib_path, **kwargs):
        instance = cls.__new__(cls)
        instance._initialize(params, result_idx, target, func_or_mod)
        text = cls._normalize_cached_text_source(device_kernel_source)
        instance.device_kernel_source = text.text if text.text is not None else Path(text.path).read_text()
        instance.libpath = kernel_lib_path
        return instance

    def get_kernel_source(self, kernel_only=False):
        return self.device_kernel_source if kernel_only else self.device_kernel_source + self.host_kernel_source

    def get_host_source(self):
        return self.host_kernel_source

    def _convert_torch_func(self):
        def invoke(*inputs):
            if len(inputs) != len(self.params) - len(self.result_idx):
                raise ValueError(f"Pokedex kernel expects {len(self.params) - len(self.result_idx)} tensor arguments")
            tensors, iterator = [], iter(inputs)
            for index, param in enumerate(self.params):
                shape = tuple(int(dim) for dim in param.shape)
                tensor = torch.empty(shape, dtype=torch.float32, device="cpu") if index in self.result_idx else next(iterator)
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ValueError("Pokedex requires CPU float32 tensors")
                if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                    raise ValueError(f"Pokedex requires a contiguous tensor with shape {shape}")
                tensors.append(tensor)
            ranges = sorted((tensor.data_ptr(), tensor.data_ptr() + size) for tensor, size in zip(tensors, self.sizes))
            if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                raise ValueError("Pokedex tensor arguments must not overlap in memory")
            executable = shutil.which(os.environ.get("TILELANG_POKEDEX", "pokedex"))
            model = os.environ.get("POKEDEX_MODEL_LIB")
            if executable is None or not model or not Path(model).is_file():
                raise RuntimeError("Set TILELANG_POKEDEX to a run-gpgpu executable and POKEDEX_MODEL_LIB to its model library")
            budget = int(os.environ.get("TILELANG_POKEDEX_MAX_INSTRUCTIONS", "10000000"))
            if budget <= 0:
                raise ValueError("TILELANG_POKEDEX_MAX_INSTRUCTIONS must be positive")
            with tempfile.TemporaryDirectory(prefix="tilelang-pokedex-call-") as directory:
                directory = Path(directory)
                memory = bytearray(self.total_size)
                for index, (tensor, offset, size) in enumerate(zip(tensors, self.offsets, self.sizes)):
                    if index not in self.result_idx:
                        memory[offset : offset + size] = tensor.detach().numpy().tobytes()
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
                with torch.no_grad():
                    for tensor, offset, size in zip(tensors, self.offsets, self.sizes):
                        values = torch.frombuffer(memory, dtype=torch.float32, count=size // 4, offset=offset)
                        tensor.copy_(values.reshape(tensor.shape))
            outputs = [tensors[index] for index in self.result_idx]
            return outputs[0] if len(outputs) == 1 else outputs if outputs else None

        return invoke
