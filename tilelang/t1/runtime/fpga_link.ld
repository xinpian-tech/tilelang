OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
  PROGRAM (RWX) : ORIGIN = 0x80000000, LENGTH = 1M
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
  /* Loader zeros every PT_LOAD BSS tail before releasing any hart.
     Mailbox: 0x81000000..0x81001000; tensors: 0x90000000..0x98000000.
     Four 8 KiB stacks: 0x9fff8000..0xa0000000. */
  __stacktop = 0xa0000000;
  ASSERT(_start == ORIGIN(PROGRAM), "FPGA entry must match boot address")
}
