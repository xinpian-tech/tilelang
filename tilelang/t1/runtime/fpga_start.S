#include "fpga_protocol.h"

    .section .text.init
    .globl _start
    .option norvc
    .option norelax
    .p2align 2
_start:
    la t0, t1_runtime_trap
    csrw mtvec, t0
    csrr a0, mhartid
    li t0, 4
    bgeu a0, t0, park
    li t0, 0x2200
    csrs mstatus, t0
    csrwi fcsr, 0
    csrwi vcsr, 0
    la sp, __stacktop
    slli t0, a0, 13
    sub sp, sp, t0
    la gp, __global_pointer$
    call t1_runtime_main
park:
    j park

    .p2align 2
t1_runtime_trap:
    csrr t0, mhartid
    li t1, 4
    bgeu t0, t1, park
    slli t1, t0, 6
    li t2, T1_MAILBOX
    add t1, t1, t2
    sw t0, T1_MB_HART(t1)
    li t2, T1_FPGA_ABI
    sw t2, T1_MB_ABI(t1)
    csrr t2, mcause
    sw t2, T1_MB_MCAUSE(t1)
    csrr t2, mepc
    sw t2, T1_MB_MEPC(t1)
    csrr t2, mtval
    sw t2, T1_MB_MTVAL(t1)
    fence iorw, iorw
    li t2, T1_STATUS_TRAP
    sw t2, T1_MB_STATUS(t1)
    fence iorw, iorw
    j park
