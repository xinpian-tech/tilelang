    .section .text.init
    .globl _start
    .p2align 2
_start:
    // Enable scalar floating point and vector state (FS=Initial, VS=Initial).
    li t0, 0x2200
    csrs mstatus, t0
    csrwi fcsr, 0
    csrwi vcsr, 0
    csrr a0, mhartid

    // The four stacks occupy disjoint 8 KiB regions at the top of SRAM.
    la sp, __stacktop
    slli t0, a0, 13
    sub sp, sp, t0
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    call demo_main

    // Returning from the harness is an error.
    li t0, 3
    la t1, tohost
    sw t0, 0(t1)
1:  j 1b
