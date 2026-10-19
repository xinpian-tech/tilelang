from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from .codegen import T1_CODEGEN
from .execution_backend import EXECUTION_BACKENDS
from .pipeline import lower

BACKEND = register_backend(
    BackendModule(
        name="t1",
        target_kinds=("t1",),
        pipelines={"t1": PassPipeline("t1", lower)},
        device_codegens={"t1": T1_CODEGEN},
        execution_backends=EXECUTION_BACKENDS,
    )
)
