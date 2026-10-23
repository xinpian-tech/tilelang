"""Synchronous four-hart FPGA execution over Linux XDMA character devices."""

import math
import os
import time

from tilelang.jit.adapter.t1 import T1KernelAdapter
from tilelang.t1 import fpga, toolchain


class XDMAKernelAdapter(T1KernelAdapter):
    runtime = "fpga"
    execution_name = "XDMA"

    def _execute(self, memory):
        # Keep Linux-specific imports out of compiler-only package imports.
        from .device import XDMADevice

        timeout = float(os.environ.get("TILELANG_XDMA_TIMEOUT", "10"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TILELANG_XDMA_TIMEOUT must be finite and positive")
        drain_delay = float(os.environ.get("TILELANG_XDMA_DRAIN_DELAY", "0"))
        if not math.isfinite(drain_delay) or drain_delay < 0:
            raise ValueError("TILELANG_XDMA_DRAIN_DELAY must be finite and nonnegative")
        entry, program = fpga.load_elf(self.libpath)
        device = os.environ.get("TILELANG_XDMA_DEVICE", "/dev/xdma0")
        vlen = int(self.target.attrs["vlen"])
        with XDMADevice(device) as board:
            board.reset()
            board.set_entry(entry)
            board.write(fpga.PROGRAM_BASE, program)
            if board.read(fpga.PROGRAM_BASE, len(program)) != program:
                raise RuntimeError("XDMA program readback mismatch")
            board.write(toolchain.DATA_BASE, memory)
            board.write(fpga.MAILBOX, bytes(fpga.PAGE_SIZE))
            if any(board.read(fpga.MAILBOX, fpga.PAGE_SIZE)):
                raise RuntimeError("XDMA mailbox clear readback mismatch")
            deadline = time.monotonic() + timeout
            board.start()
            done = []
            while time.monotonic() < deadline:
                board.check_faults()
                mailbox = board.read(fpga.MAILBOX, fpga.PAGE_SIZE)
                # A DMA burst may start before a hart's publication store.
                # Fetch again after seeing any status before consuming fields.
                if any(mailbox[h * fpga.PROTOCOL["T1_RECORD_SIZE"] : h * fpga.PROTOCOL["T1_RECORD_SIZE"] + 4] != b"\0" * 4 for h in range(4)):
                    mailbox = board.read(fpga.MAILBOX, fpga.PAGE_SIZE)
                done = fpga.completion(mailbox, vlen)
                if len(done) == 4:
                    # The scalar mailbox store waits for prior RVV/VME stores
                    # to retire. Retain an opt-in delay only for bring-up/debug.
                    if drain_delay:
                        time.sleep(drain_delay)
                    board.check_faults()
                    output = board.read(toolchain.DATA_BASE, len(memory))
                    board.check_faults()
                    return output
                time.sleep(0.001)
            raise RuntimeError(f"XDMA timed out after {timeout:g}s; completed harts: {done}")
