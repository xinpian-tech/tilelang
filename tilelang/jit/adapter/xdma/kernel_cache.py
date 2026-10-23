from tilelang.cache.kernel_cache import KernelCache
from tilelang.t1.toolchain import cache_identity


class XDMAKernelCache(KernelCache):
    @staticmethod
    def _get_base_key():
        return {**KernelCache._get_base_key(), "xdma_toolchain": cache_identity()}
