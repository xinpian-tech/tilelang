    .section .text.init
    .globl _start
    .p2align 2
_start:
    li t0, 0x2200
    csrs mstatus, t0
    csrwi fcsr, 0
    csrwi vcsr, 0
    la t0, t1_runtime_trap
    csrw mtvec, t0
    csrr a0, mhartid
    la sp, __stacktop
    slli t0, a0, 13
    sub sp, sp, t0
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    call t1_runtime_main
    .p2align 2
t1_runtime_trap:
    li t0, 3
    la t1, tohost
    sw t0, 0(t1)
1:  j 1b
