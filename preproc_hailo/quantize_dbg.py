# quantize.py with a probe: print every shortcut insertion in the mixed-precision pass
import runpy, sys
from hailo_model_optimization.algorithms.mixed_precision import create_mixed_precision as M
from hailo_model_optimization.acceleras.utils.acceleras_definitions import PrecisionMode
cls = [c for c in vars(M).values() if isinstance(c, type) and hasattr(c, "_add_shortcut_layer")][0]
orig = cls._add_shortcut_layer
def probe(self, source, targets, mode):
    sucs = self._model.flow.successors_sorted(source); cfg = self.get_algo_config()
    print("SHORTCUT", source, "targets", targets, "mode", mode, "succ", [(s, str(cfg.layers[s].precision_mode)) for s in sucs], flush=True)
    return orig(self, source, targets, mode)
cls._add_shortcut_layer = probe
sys.argv = ["quantize.py"] + sys.argv[1:]; runpy.run_path("quantize.py", run_name="__main__")
