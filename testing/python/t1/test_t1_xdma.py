"""FPGA ABI/transport tests; board tests opt in with TILELANG_TEST_XDMA=1."""

import fcntl
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

import tilelang
from tilelang.backend import create_backend_context
from tilelang.t1 import fpga, toolchain
from tilelang.jit.adapter.xdma.device import XDMADevice
from examples.t1.vector_add import vector_add
from testing.python.t1.test_t1_backend import CODEGENS


def elf_bytes(*, address=0x80000000, filesz=4, memsz=16, entry=0x80000000):
    ident = b"\x7fELF\x01\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIIIIIHHHHHH", ident, 2, 243, 1, entry, 52, 0, 0, 52, 32, 1, 0, 0, 0)
    segment = struct.pack("<8I", 1, 84, address, address, filesz, memsz, 5, 4)
    return header + segment + b"\x13\0\0\0"


def test_elf_zero_fill_and_validation(tmp_path):
    path = tmp_path / "kernel.elf"
    path.write_bytes(elf_bytes())
    entry, image = fpga.load_elf(path)
    assert entry == 0x80000000
    assert image[:4] == b"\x13\0\0\0"
    assert not any(image[4:])
    for data, error in [
        (b"short", "ELF32"),
        (elf_bytes(address=0x81000000), "reservation"),
        (elf_bytes(filesz=20), "sizes"),
        (elf_bytes(memsz=2), "sizes"),
        (elf_bytes(entry=0x80000008), "entry"),
    ]:
        path.write_bytes(data)
        with pytest.raises(ValueError, match=error):
            fpga.load_elf(path)


def test_mailbox_waits_for_four_harts_and_reports_traps():
    mailbox = bytearray(4096)
    for hart in range(4):
        assert fpga.completion(mailbox, 1024) == list(range(hart))
        struct.pack_into("<7I", mailbox, hart * 64, 0x50415353, 1, hart, 0, 0, 0, 128)
    assert fpga.completion(mailbox, 1024) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="hardware=1024, compiled=256"):
        fpga.completion(mailbox, 256)
    struct.pack_into("<7I", mailbox, 2 * 64, 0x54524150, 1, 2, 2, 0x80000100, 0xFFFFFFFF, 0)
    with pytest.raises(RuntimeError, match="hart=2 mcause=0x00000002 mepc=0x80000100 mtval=0xffffffff"):
        fpga.completion(mailbox, 1024)


def make_file_device(tmp_path):
    prefix = str(tmp_path / "xdma0")
    for suffix in ("_user", "_h2c_0"):
        with open(prefix + suffix, "wb") as file:
            file.truncate(2 * 1024 * 1024)
    os.link(prefix + "_h2c_0", prefix + "_c2h_0")
    with open(prefix + "_user", "r+b") as file:
        for base in (0x1000, 0x2000):
            file.seek(base + 0x10)
            file.write(struct.pack("<I", 2))
    return prefix


class FirewallDevice(XDMADevice):
    """File-backed DMA with register handshakes and reset-order assertions."""

    def __init__(self, prefix):
        super().__init__(prefix)
        self.events = []
        self.status = [0, 0]
        self.polls = [0, 0]
        self.forced = 0
        self.stall_drain = False
        self.stall_unblock = False
        self.block_status = self.force_both

    def _read_register(self, offset):
        if offset in (0x1100, 0x2100):
            index = (offset - 0x1100) // 0x1000
            if self.status[index] & self.busy and not self.stall_drain:
                assert self.forced == 3, "both force requests must precede drain polling"
                self.polls[index] += 1
                if self.polls[index] >= 2 + index * 2:
                    self.status[index] &= ~self.busy
            return self.status[index]
        return super()._read_register(offset)

    def _write_register(self, offset, value):
        self.events.append((offset, value))
        if offset == 0:
            if value == 0:
                assert all(self._blocked_both(s, 0) and not s & self.busy for s in self.status)
            else:
                assert self.status == [0, 0]
        for index, base in enumerate(self.firewall_bases):
            if offset == base + self.firewall_force:
                assert value in (0, self.force_both)
                if value:
                    self.forced |= 1 << index
                    self.polls[index] = 0
                    self.status[index] = self.block_status | self.busy
            if offset == base + self.firewall_unblock:
                assert super()._read_register(0) == 0, "unblock must happen under reset"
                assert super()._read_register(base + self.firewall_force) == 0, "clear pending force before unblock"
                self.forced &= ~(1 << index)
                if not self.stall_unblock or index == 0:
                    self.status[index] = 0
        super()._write_register(offset, value)


def test_dma_padding(tmp_path):
    prefix = make_file_device(tmp_path)
    payload = bytearray((i % 251 for i in range(1024 * 1024 + 13)))
    with FirewallDevice(prefix) as board:
        board.reset()
        board.set_entry(0x80000000)
        board.write(0, payload)
        assert board.read(0, len(payload)) == payload
        assert Path(prefix + "_h2c_0").read_bytes()[len(payload) : len(payload) + 4096 - 13] == bytes(4096 - 13)


def test_device_exclusion_and_cleanup(tmp_path):
    prefix = make_file_device(tmp_path)
    with pytest.raises(RuntimeError, match="guest failure"):
        with FirewallDevice(prefix) as board:
            board.reset()
            board.start()
            with pytest.raises(RuntimeError, match="already in use"):
                with XDMADevice(prefix):
                    pass
            os.truncate(prefix + "_c2h_0", 1)
            with pytest.raises(RuntimeError, match="short read"):
                board.read(0, 4096)
            raise RuntimeError("guest failure")
    assert struct.unpack_from("<I", Path(prefix + "_user").read_bytes())[0] == 0
    with open(prefix + "_user", "rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_firewall_launch_and_cleanup_order(tmp_path):
    with FirewallDevice(make_file_device(tmp_path)) as board:
        with pytest.raises(RuntimeError, match="completed isolation"):
            board.start()
        board.reset()
        board.set_entry(0x80000000)
        board.start()
        board.check_faults()
    assert board.events == [
        (0x1104, 0x08000080),
        (0x2104, 0x08000080),
        (0, 0),
        (8, 0x80000000),
        (0x1104, 0),
        (0x2104, 0),
        (0x1108, 1),
        (0x2108, 1),
        (0, 1),
        (0x1104, 0x08000080),
        (0x2104, 0x08000080),
        (0, 0),
    ]
    assert board.isolated and board.held and not board.fds


def test_firewall_version_mismatch_does_not_write(tmp_path):
    prefix = make_file_device(tmp_path)
    with open(prefix + "_user", "r+b") as file:
        file.seek(0x2010)
        file.write(struct.pack("<I", 1))
    device = FirewallDevice(prefix)
    with pytest.raises(RuntimeError, match="refusing control writes"):
        with device:
            pass
    assert not device.events and not device.fds


def test_firewall_drain_timeout_never_resets_or_unblocks(tmp_path):
    device = FirewallDevice(make_file_device(tmp_path))
    with pytest.raises(RuntimeError, match="drain timed out.*no reset/unblock"):
        with device as board:
            board.stall_drain = True
            board.recovery_timeout = 0
            board.reset()
    assert device.events == [(0x1104, 0x08000080), (0x2104, 0x08000080)]
    assert device.drain_failed and not device.fds


def test_firewall_partial_unblock_is_isolated_again(tmp_path):
    device = FirewallDevice(make_file_device(tmp_path))
    with pytest.raises(RuntimeError, match="unblock timed out.*tile remains reset"):
        with device as board:
            board.reset()
            board.stall_unblock = True
            board.recovery_timeout = 0.1
            board.start()
    assert (0, 1) not in device.events
    assert device.events[-3:] == [(0x1104, 0x08000080), (0x2104, 0x08000080), (0, 0)]
    assert device.isolated and device.held


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_firewall_fault_detection_and_recovery(tmp_path, during_cleanup):
    device = FirewallDevice(make_file_device(tmp_path))
    with pytest.raises(RuntimeError, match="AXI firewall fault"):
        with device as board:
            # Existing automatic blocking faults are recoverable at entry.
            board.block_status = 0x00020002
            board.reset()
            board.start()
            if not during_cleanup:
                board.status[1] = 0x00000020  # Report-only SLVERR is still a failure.
                board.check_faults()
            # Otherwise an automatic fault arises while exit isolates the tile.
    assert device.held and device.isolated and not device.fds


@pytest.mark.parametrize(
    ("status", "enables", "blocked"), [(0x00200020, 0, False), (0x02000020, 0, False), (0x02000020, 0x40004000, True)]
)
def test_firewall_response_error_block_enables(status, enables, blocked):
    assert XDMADevice._blocked_both(status, enables) is blocked


@pytest.mark.parametrize("delay", [None, "0", "0.25"])
def test_completion_delay_before_output_and_reset(tmp_path, monkeypatch, delay):
    from tilelang.jit.adapter.xdma import adapter
    from tilelang.jit.adapter.xdma import device

    path = tmp_path / "kernel.elf"
    path.write_bytes(elf_bytes())
    events = []

    class Board:
        def __init__(self, prefix):
            self.memory = {}
            self.running = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("reset")

        def reset(self):
            pass

        def set_entry(self, entry):
            pass

        def check_faults(self):
            pass

        def write(self, address, data):
            self.memory[address] = bytearray(data)

        def start(self):
            self.running = True
            for hart in range(4):
                struct.pack_into("<7I", self.memory[0x81000000], hart * 64, 0x50415353, 1, hart, 0, 0, 0, 128)

        def read(self, address, size):
            if self.running and address == 0x90000000:
                events.append("output")
            return self.memory[address][:size]

    monkeypatch.setattr(device, "XDMADevice", Board)
    monkeypatch.setattr(adapter.time, "sleep", lambda delay: events.append(("sleep", delay)))
    monkeypatch.delenv("TILELANG_XDMA_DRAIN_DELAY", raising=False)
    if delay is not None:
        monkeypatch.setenv("TILELANG_XDMA_DRAIN_DELAY", delay)
    kernel = adapter.XDMAKernelAdapter.__new__(adapter.XDMAKernelAdapter)
    kernel.libpath = str(path)
    kernel.target = create_backend_context({"kind": "t1", "vlen": 1024}, None, "xdma").target
    assert kernel._execute(bytearray(16)) == bytearray(16)
    assert events == ([("sleep", 0.25)] if delay == "0.25" else []) + ["output", "reset"]


@pytest.mark.parametrize("codegen", CODEGENS)
def test_fpga_build_and_cache_without_device(tmp_path, codegen):
    script = tmp_path / "probe.py"
    script.write_text('''import os
import tilelang
from examples.t1.vector_add import vector_add
from tilelang.t1 import fpga, toolchain
if os.environ.get("CACHE_MUST_HIT"):
    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not build")
    toolchain.build = fail
k = tilelang.compile(vector_add(131), target={"kind":"t1", "vlen":1024, "codegen":os.environ["CODEGEN"]}, execution_backend="xdma", out_idx=2)
fpga.load_elf(k.adapter.libpath)
assert "vlenb" in k.get_host_source()
assert k.get_kernel_source()
''')
    environment = {**os.environ, "TILELANG_CACHE_DIR": str(tmp_path / "cache"), "CODEGEN": codegen}
    for cached in (False, True):
        if cached:
            environment["CACHE_MUST_HIT"] = "1"
        result = subprocess.run([sys.executable, str(script)], env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


def test_xdma_selection_and_failure_preserves_tensors(monkeypatch):
    assert create_backend_context("t1", None, "auto").execution_backend.name == "pokedex"
    assert create_backend_context("t1", None, "xdma").execution_backend.name == "xdma"
    kernel = tilelang.compile(vector_add(4), target="t1", execution_backend="xdma")
    a, b, c = torch.ones(4), torch.ones(4), torch.full((4,), 17.0)
    monkeypatch.setenv("TILELANG_XDMA_DEVICE", "/missing/xdma0")
    with pytest.raises(RuntimeError, match="device .* is missing"):
        kernel(a, b, c)
    torch.testing.assert_close(c, torch.full((4,), 17.0), rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="profiling"):
        kernel.get_profiler()
    with pytest.raises(NotImplementedError, match="bare-metal ELF"):
        kernel.export_library("unused.so")


def test_multicall_tool_keeps_invocation_name(tmp_path, monkeypatch):
    driver = tmp_path / "lld"
    driver.write_text("#!/bin/sh\nexit 0\n")
    driver.chmod(0o755)
    (tmp_path / "ld.lld").symlink_to(driver)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert toolchain.required_tool("ld.lld") == str(tmp_path / "ld.lld")


@pytest.fixture
def board(monkeypatch):
    if os.environ.get("TILELANG_TEST_XDMA") != "1":
        pytest.skip("set TILELANG_TEST_XDMA=1 to reset/run the t1_demo FPGA")
    monkeypatch.setenv("TILELANG_XDMA_DRAIN_DELAY", "0")


@pytest.mark.parametrize("codegen", CODEGENS)
@pytest.mark.parametrize("n", [3, 259, 1025, 65539])
def test_board_vector_add(board, codegen, n):
    kernel = tilelang.compile(vector_add(n), target={"kind": "t1", "vlen": 1024, "codegen": codegen}, execution_backend="xdma", out_idx=2)
    for seed in (range(10) if n == 65539 else (7, 19)):
        generator = torch.Generator().manual_seed(seed)
        a, b = (torch.randn(n, generator=generator) for _ in range(2))
        torch.testing.assert_close(kernel(a, b), a + b, rtol=0, atol=0)


def test_board_vlen_mismatch(board):
    kernel = tilelang.compile(vector_add(4), target={"kind": "t1", "vlen": 256}, execution_backend="xdma")
    a, b, c = torch.ones(4), torch.ones(4), torch.full((4,), 17.0)
    with pytest.raises(RuntimeError, match="VLEN mismatch"):
        kernel(a, b, c)
    torch.testing.assert_close(c, torch.full((4,), 17.0), rtol=0, atol=0)


@pytest.mark.parametrize("codegen", CODEGENS)
def test_board_timeout_then_restart(board, monkeypatch, codegen):
    kernel = tilelang.compile(vector_add(65539), target={"kind": "t1", "vlen": 1024, "codegen": codegen}, execution_backend="xdma")
    a, b, c = torch.ones(65539), torch.ones(65539), torch.full((65539,), 17.0)
    monkeypatch.setenv("TILELANG_XDMA_TIMEOUT", "0.000001")
    with pytest.raises(RuntimeError, match="XDMA timed out"):
        kernel(a, b, c)
    torch.testing.assert_close(c, torch.full_like(c, 17.0), rtol=0, atol=0)
    monkeypatch.setenv("TILELANG_XDMA_TIMEOUT", "10")
    kernel(a, b, c)
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)
