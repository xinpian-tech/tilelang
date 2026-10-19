OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
  PROGRAM (RWX) : ORIGIN = 0x80000000, LENGTH = 256M
}
SECTIONS {
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > PROGRAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > PROGRAM
  .data : { *(.data .data.*) } > PROGRAM
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > PROGRAM
  .bss (NOLOAD) : { *(.bss .bss.* .sbss .sbss.* COMMON) } > PROGRAM
  /* 0x90000000..0x98000000 is reserved for file-based tensor exchange. */
  __stacktop = 0xa0000000;
  fromhost = 0x01000000;
  tohost = 0x01000008;
}
