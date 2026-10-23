"""ELF and mailbox contract for the four-hart t1_demo FPGA image."""

from pathlib import Path
import re
import struct

from .toolchain import RUNTIME

# Share the wire layout with the guest assembly and C++ launcher.
PROTOCOL = {
    name: int(value, 0)
    for name, value in re.findall(r"^#define (T1_\w+) (0x[0-9a-fA-F]+|\d+)$", (RUNTIME / "fpga_protocol.h").read_text(), re.M)
}
PROGRAM_BASE = 0x80000000
PROGRAM_SIZE = 0x100000
MAILBOX = PROTOCOL["T1_MAILBOX"]
PAGE_SIZE = 4096


def load_elf(path):
    """Validate before touching hardware; return entry and a zero-filled image.

    Only fixed-address little-endian RV32 executables in the program reservation
    are accepted. Loading one image also clears BSS and holes on repeated calls.
    """
    data = Path(path).read_bytes()
    if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("XDMA requires a little-endian ELF32 executable")
    _, kind, machine, version, entry, phoff, _, _, ehsize, phsize, count, _, _, _ = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    if (kind, machine, version, ehsize, phsize) != (2, 243, 1, 52, 32):
        raise ValueError("XDMA requires an RV32 ET_EXEC ELF with standard program headers")
    if not count or phoff < ehsize or phoff + count * phsize > len(data):
        raise ValueError("XDMA ELF has a truncated or missing program-header table")
    segments = []
    entry_valid = False
    for index in range(count):
        typ, offset, vaddr, paddr, filesz, memsz, flags, align = struct.unpack_from("<8I", data, phoff + index * phsize)
        if typ in (2, 3):
            raise ValueError("XDMA cannot load a dynamic ELF")
        if typ != 1:
            continue
        if filesz > memsz or offset + filesz > len(data):
            raise ValueError("XDMA ELF segment has invalid file/memory sizes")
        if vaddr != paddr or paddr < PROGRAM_BASE or paddr + memsz > PROGRAM_BASE + PROGRAM_SIZE:
            raise ValueError("XDMA ELF segment is outside the 1 MiB program reservation")
        if align > 1 and (align & (align - 1) or (vaddr - offset) % align):
            raise ValueError("XDMA ELF segment has invalid alignment")
        if memsz:
            segments.append((paddr, paddr + memsz, offset, filesz))
            entry_valid |= bool(flags & 1) and paddr <= entry < paddr + filesz
    if entry != PROGRAM_BASE or not entry_valid:
        raise ValueError("XDMA ELF entry must be executable code at 0x80000000")
    segments.sort()
    if any(a[1] > b[0] for a, b in zip(segments, segments[1:])):
        raise ValueError("XDMA ELF load segments overlap")
    image = bytearray(PROGRAM_SIZE)
    for start, _, offset, size in segments:
        image[start - PROGRAM_BASE : start - PROGRAM_BASE + size] = data[offset : offset + size]
    return entry, image


def completion(mailbox, vlen):
    """Return completed hart IDs, or raise with a specific guest diagnostic."""
    done = []
    for hart in range(4):
        record = hart * PROTOCOL["T1_RECORD_SIZE"]

        def word(name):
            return struct.unpack_from("<I", mailbox, record + PROTOCOL[f"T1_MB_{name}"])[0]

        status = word("STATUS")
        if not status:
            continue
        if word("ABI") != PROTOCOL["T1_FPGA_ABI"] or word("HART") != hart:
            raise RuntimeError(f"XDMA mailbox ABI/hart mismatch in record {hart}")
        if status == PROTOCOL["T1_STATUS_TRAP"]:
            raise RuntimeError(
                f"XDMA guest trap: hart={hart} mcause=0x{word('MCAUSE'):08x} "
                f"mepc=0x{word('MEPC'):08x} mtval=0x{word('MTVAL'):08x}"
            )
        if status == PROTOCOL["T1_STATUS_CONFIG"] or word("VLENB") * 8 != vlen:
            raise RuntimeError(f"XDMA VLEN mismatch: hart={hart}, hardware={word('VLENB') * 8}, compiled={vlen}")
        if status != PROTOCOL["T1_STATUS_PASS"]:
            raise RuntimeError(f"XDMA unknown mailbox status 0x{status:08x} for hart {hart}")
        done.append(hart)
    return done
