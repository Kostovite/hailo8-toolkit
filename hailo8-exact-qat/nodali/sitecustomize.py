"""Hide NVIDIA DALI from the DFC: its cuda120 2.2.0 build fails on Maxwell (CUDA_ERROR_INVALID_VALUE in the MIXED
MakeContiguous op), which kills AdaRound. hailo_model_optimization.algorithms.dali_utils then falls back to its TF
dataset path. Active only with PYTHONPATH=.../hailo_re/nodali."""
import importlib.abc
import sys


class _NoDali(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "nvidia.dali" or name.startswith("nvidia.dali."):
            raise ModuleNotFoundError(f"{name} disabled on Maxwell (hailo_re/nodali)", name=name)
        return None


sys.meta_path.insert(0, _NoDali())
