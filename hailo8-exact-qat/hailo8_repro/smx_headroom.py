"""Softmax-denominator headroom (RULES.md section 26). The DFC sizes the 16-bit WRAPPING accumulator of every
decomposed softmax's reduce_sum from the calibration maximum of its pre-activation (-sum of exp), so any row whose
exp-sum exceeds the calibrated maximum wraps around and gets a reciprocal of ~0 (the attention row is zeroed).
import smx_headroom; smx_headroom.install(mode) before ClientRunner.optimize():
  mode "full": size it for the worst case (reduce size x max exp input x |kernel|) -> can never wrap
  mode <float f>: f x the calibrated range (capped at the worst case)
"""
import numpy as np

_installed = {}


def install(mode="full", log=print):
    from hailo_model_optimization.acceleras.hailo_layers.hailo_reduce_sum import HailoReduceSum
    if "orig" not in _installed:
        _installed["orig"] = HailoReduceSum.create_hw_params
    orig = _installed["orig"]

    def create_hw_params(self, weights_clipping, optimization_target, hw_shifts=None):
        name = getattr(self, "full_name", "") or getattr(self, "name", "")
        if "reduce_sum_softmax" not in name:
            return orig(self, weights_clipping, optimization_target, hw_shifts=hw_shifts)
        act = self.act_op
        st = act.get_input_stats(0)
        in_max = float(np.max(np.abs(self.reduce_sum_op.get_input_limvals(0))))
        kmax = float(np.max(np.abs(np.asarray(self.reduce_sum_op.kernel))))
        worst = float(self.reduce_sum_op._reduce_size) * in_max * kmax
        cal = float(np.max(np.maximum(np.abs(st.min), np.abs(st.max))))
        tgt = worst if mode == "full" else min(worst, float(mode) * cal)
        if tgt > cal:
            new = st._replace(min=np.minimum(st.min, -tgt))  # only the far end: a row sum is >= 1 (max exp = 1)
            act.get_input_stats = lambda i, _s=new, _g=act.get_input_stats: _s if i == 0 else _g(i)
            log(f"smx_headroom: {name} denominator range {cal:.1f} -> {tgt:.1f} (worst case {worst:.1f})")
        try:
            return orig(self, weights_clipping, optimization_target, hw_shifts=hw_shifts)
        finally:
            act.__dict__.pop("get_input_stats", None)

    HailoReduceSum.create_hw_params = create_hw_params
