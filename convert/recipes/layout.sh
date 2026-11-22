# PP-DocLayoutV3 (r5_ord5 family) chip part. Host: enc_output, score/box heads, top-k, deformable decoder, mask + order heads.
# Rewrites (all float-exact, checked): onnxsim -> compat (MatMulAdd fusion OFF) -> 3 qkv identity matmuls -> bilinear x2 resizes
# = nearest x2 + concat-slice edge pad + dw3x3 [.25,.5,.25]  (Hailo: 16-bit resize output not compilable; edge Pad silently
# becomes zero pad; grouped deconv unsupported). Precision: mix16 (+ LAYOUT_8BIT filter of the upsample blocks via gen8).
QMODE=mix16; NORM=rgb255; OPT=0; INPUT16=0; CHIP_FMT=uint8; FP_TOL=${FP_TOL:-2e-2}; FP_ROWS=32; export EMU_F16=1
LAYOUT_8BIT=${LAYOUT_8BIT-_Resize}        # "" = keep upsample blocks 16-bit; "_encoder_Resize_2" = only the 224 one
r_export(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/layout_torch.py export $CKPT $OUT; }   # needs transformers 5.22 (pp_doclayout_v3)
r_prep(){ tenv; export PYTHONPATH=$QAT/tools/ops:${PYTHONPATH:-}; set -e; cd $OUT
  python -u -m onnxsim model.onnx model.sim.onnx
  python -u $CV/recipes/onnx_hailo_compat_nofuse.py model.sim.onnx model.compat.onnx
  python -u $PP/add_qkv_identity.py model.compat.onnx model.id.onnx
  python -c "import onnx;n=[x for x in onnx.load('model.id.onnx').graph.node if '/IdMatMul' in x.name];print('QKV_IDENTITY',len(n));assert len(n)==3"
  ( henv; "$HPY" -u $CV/recipes/rewrite_resize.py model.id.onnx model.up.onnx nn && OMP_NUM_THREADS=16 "$HPY" $CV/recipes/onnx_parity.py model.id.onnx model.up.onnx )
  mv model.onnx model.raw.onnx; cp model.up.onnx model.onnx; }
r_data(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/layout_torch.py data $CKPT $OUT && r_gate ref; }
r_alls(){ henv; export CUDA_VISIBLE_DEVICES=99; [ -z "$LAYOUT_8BIT" ] && { : > $OUT/extra.alls; return 0; }
  "$HPY" -u $CV/recipes/gen8.py $OUT/parsed.har $OUT/extra.alls $LAYOUT_8BIT; }
EXTRA_ALLS=$OUT/extra.alls
r_gate(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/layout_gate.py $CKPT $OUT $1; }
