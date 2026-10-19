from tilelang.backend.execution_backend import ExecutionBackendSpec

# Source generation needs no compiler/emulator installation. The adapter checks
# the tools when building/loading an executable, with actionable errors.
EXECUTION_BACKENDS = (ExecutionBackendSpec("pokedex"),)
