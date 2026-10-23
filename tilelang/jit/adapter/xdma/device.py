"""Aligned DMA and firewall-protected reset ownership for a t1_demo board."""

import ctypes
import fcntl
import mmap
import os
import time


class XDMADevice:
    alignment = 4096
    chunk_size = 1024 * 1024
    bar_size = 0x3000
    firewall_bases = (0x1000, 0x2000)
    firewall_version = 0x010
    firewall_status = 0x100
    firewall_force = 0x104
    firewall_unblock = 0x108
    firewall_enables = 0x208
    force_both = 0x08000080
    busy = 0x00010001
    faults = 0x0FFE00FE
    recovery_timeout = 1.0

    def __init__(self, prefix):
        self.prefix = prefix
        self.fds = []
        self.gpio = None
        self.armed = False
        self.isolated = False
        self.held = False
        self.running = False
        self.drain_failed = False

    def __enter__(self):
        try:
            self.user = self._open("_user", os.O_RDWR | os.O_SYNC)
            try:
                fcntl.flock(self.user, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"XDMA board {self.prefix} is already in use") from error
            self.h2c = self._open("_h2c_0", os.O_WRONLY)
            self.c2h = self._open("_c2h_0", os.O_RDONLY)
            self.gpio = mmap.mmap(self.user, self.bar_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            versions = [self._read_register(base + self.firewall_version) for base in self.firewall_bases]
            if versions != [2, 2]:
                raise RuntimeError(f"XDMA requires AXI firewall version 2 at 0x1000 and 0x2000; got {versions}; refusing control writes")
            self.armed = True
            return self
        except BaseException:
            self._close()
            raise

    def _open(self, suffix, flags):
        path = self.prefix + suffix
        try:
            fd = os.open(path, flags | os.O_CLOEXEC)
        except PermissionError as error:
            raise RuntimeError(f"XDMA cannot access {path}; configure the xdma group/udev rule described in examples/t1/README.md") from error
        except FileNotFoundError as error:
            raise RuntimeError(f"XDMA device {path} is missing; load the driver or set TILELANG_XDMA_DEVICE") from error
        self.fds.append(fd)
        return fd

    def _read_register(self, offset):
        return ctypes.c_uint32.from_buffer(self.gpio, offset).value

    def _write_register(self, offset, value):
        # Control registers require one aligned 32-bit MMIO access, including
        # side-effect registers such as UNBLOCK.
        ctypes.c_uint32.from_buffer(self.gpio, offset).value = value

    def _gpio_write(self, offset, value, mask):
        self._write_register(offset, value)
        actual = self._read_register(offset)
        if (actual & mask) != (value & mask):
            raise RuntimeError(f"XDMA GPIO readback failed at +0x{offset:x}: wrote 0x{value:x}, read 0x{actual:x}")

    def _status(self):
        return [self._read_register(base + self.firewall_status) for base in self.firewall_bases]

    @staticmethod
    def _format_status(status):
        return f"scalar=0x{status[0]:08x}, vector=0x{status[1]:08x}"

    @staticmethod
    def _blocked_both(status, enables):
        # SLVERR/DECERR can be report-only; count them as blocking only when
        # their corresponding block-enable bits are set (t1_demo runner ABI).
        read = status & 0x0000009E or (status & (1 << 5) and enables & (1 << 14)) or (status & (1 << 6) and enables & (1 << 15))
        write = status & 0x09FE0000 or (status & (1 << 25) and enables & (1 << 30)) or (status & (1 << 26) and enables & (1 << 31))
        return bool(read and write)

    def check_faults(self):
        status = self._status()
        if any(value & self.faults for value in status):
            raise RuntimeError(f"XDMA AXI firewall fault/block: {self._format_status(status)}")

    def _isolate(self):
        if not self.armed:
            raise RuntimeError("XDMA firewall versions must be validated before isolation")
        if self.drain_failed:
            raise RuntimeError("XDMA firewall drain previously timed out; leaving blocks requested; no reset/unblock")
        before = self._status()
        if self.isolated:
            return before, before
        enables = [self._read_register(base + self.firewall_enables) for base in self.firewall_bases]
        # Request BOTH before polling either: resetting T1/IC0 with outstanding
        # downstream transactions can strand the shared DDR/XDMA path.
        for base in self.firewall_bases:
            self._write_register(base + self.firewall_force, self.force_both)
        deadline = time.monotonic() + self.recovery_timeout
        while True:
            status = self._status()  # Readback also completes posted MMIO writes.
            if all(self._blocked_both(value, enable) and not value & self.busy for value, enable in zip(status, enables)):
                self.isolated = True
                return before, status
            if time.monotonic() >= deadline:
                self.drain_failed = True
                raise RuntimeError(
                    f"XDMA firewall drain timed out: {self._format_status(status)}; leaving blocks requested; no reset/unblock"
                )
            time.sleep(0.00005)

    def reset(self, *, strict=False):
        before, status = self._isolate()
        self._gpio_write(0, 0, 1)
        self.held = True
        self.running = False
        time.sleep(0.001)
        if strict and (any(value & self.faults for value in before) or status != [self.force_both, self.force_both]):
            raise RuntimeError(
                f"XDMA AXI firewall fault during isolation: before {self._format_status(before)}; after {self._format_status(status)}"
            )

    def set_entry(self, entry):
        if not self.held or not self.isolated:
            raise RuntimeError("XDMA reset vector requires an isolated tile held in reset")
        self._gpio_write(8, entry, 0xFFFFFFFF)
        time.sleep(0.001)

    def start(self):
        if not self.held or not self.isolated or self.drain_failed:
            raise RuntimeError("XDMA start requires completed isolation and a tile held in reset")
        # A partial unblock invalidates isolation. Cleanup must block both again.
        self.isolated = False
        # FORCE is level-sensitive. Reasserting it on an already blocked
        # firewall leaves a pending request (auto-clear needs a new fault edge).
        # Clear it before UNBLOCK, or the firewall immediately blocks again.
        for base in self.firewall_bases:
            self._write_register(base + self.firewall_force, 0)
        for base in self.firewall_bases:
            self._write_register(base + self.firewall_unblock, 1)
        deadline = time.monotonic() + self.recovery_timeout
        while True:
            status = self._status()
            if status == [0, 0]:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"XDMA firewall unblock timed out: {self._format_status(status)}; tile remains reset")
            time.sleep(0.00005)
        # Treat a failed release readback as possibly running for cleanup.
        self.held = False
        self.running = True
        self._gpio_write(0, 1, 1)

    def write(self, address, data):
        self._transfer(address, len(data), data)

    def read(self, address, size):
        return self._transfer(address, size, None)

    def _transfer(self, address, size, data):
        if address % self.alignment or size <= 0:
            raise ValueError("XDMA transfers require a page-aligned address and positive size")
        padded = (size + self.alignment - 1) // self.alignment * self.alignment
        output = bytearray() if data is None else None
        # Anonymous mmap supplies a page-aligned user buffer. Padding stays
        # within the caller's reserved region; never pad individual ELF segments.
        with mmap.mmap(-1, min(padded, self.chunk_size)) as buffer:
            for offset in range(0, padded, self.chunk_size):
                count = min(padded - offset, self.chunk_size)
                actual = min(size - offset, count)
                if data is not None:
                    buffer[:actual] = data[offset : offset + actual]
                    buffer[actual:count] = b"\0" * (count - actual)
                # Requires the patched XDMA iterator handlers; see examples/t1/README.md.
                with memoryview(buffer)[:count] as view:
                    if data is None:
                        transferred = os.preadv(self.c2h, [view], address + offset)
                    else:
                        transferred = os.pwritev(self.h2c, [view], address + offset)
                if transferred != count:
                    raise RuntimeError(f"XDMA short {'read' if data is None else 'write'} at 0x{address + offset:x}: {transferred}/{count}")
                if output is not None:
                    output.extend(buffer[:actual])
        return output

    def _close(self):
        if self.gpio is not None:
            self.gpio.close()
            self.gpio = None
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.armed and not self.drain_failed:
                self.reset(strict=self.running and exc_type is None)
        finally:
            self._close()
