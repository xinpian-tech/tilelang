OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  SRAM (RWX) : ORIGIN = 0x80000000, LENGTH = 512M
}

SECTIONS {
  .text : {
    KEEP(*(.text.init))
    *(.text .text.*)
  } > SRAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > SRAM
  .data : { *(.data .data.*) } > SRAM
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > SRAM
  .bss (NOLOAD) : {
    *(.bss .bss.* .sbss .sbss.* COMMON)
  } > SRAM
  __stacktop = ORIGIN(SRAM) + LENGTH(SRAM);
  ASSERT(. <= __stacktop - 4 * 8192, "Program overlaps four hart stacks")
  fromhost = 0x01000000;
  tohost = 0x01000008;
}
