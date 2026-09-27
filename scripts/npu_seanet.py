#!/usr/bin/env python3
"""Incremental SEANet -> Google Tensor NPU export probe.

The full SEANet decoder (`pt_mimi_deconly_fp16.tflite`) partitions cleanly for
the Tensor G5 but the backend compiler then dies with an opaque INTERNAL error
(the ``compiler_worker`` segfaults, so there is no op-level diagnostic). This
script narrows the crashing op/pattern down by exporting and AOT-compiling the
decoder *prefix* one plan step at a time.

Two subcommands, because they need different venvs:

  export   torch + litert_torch   (the "GPU export" path) -> a .tflite
  aot      ai_edge_litert.aot + ai-edge-litert-sdk-google-tensor -> *_g5.tflite

Typical sweep (see scripts/npu_seanet.sh):

  ~/pockettts-conv/.venv/bin/python scripts/npu_seanet.py export 5
  ~/pockettts-aot/.venv/bin/python  scripts/npu_seanet.py aot <file>

Plan steps of the MimiDecOnly / SEANet decoder (window 4096):

   0 conv 512->512 k7        14 elu
   1 elu                     15 conv 64->128 k1   (resnet)
   2 zs  512->256 s6         16 res_close
   3 res_open                17 elu
   4 elu                     18 zs  128->64 s4
   5 conv 256->128 k3        19 res_open
   6 elu                     20 elu
   7 conv 128->256 k1        21 conv 64->32 k3
   8 res_close               22 elu
   9 elu                     23 conv 32->64 k1
  10 zs  256->128 s5         24 res_close
  11 res_open                25 elu
  12 elu                     26 conv 64->1 k3
  13 conv 128->64 k3

``zs`` is ZeroStuffConvT1d: nearest-neighbour interpolate + zero-stuff mask +
grouped conv1d, the streaming-exact ConvTranspose replacement.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PT_OUT", os.path.join(HERE, "out"))
PROBE = os.path.join(OUT, "npu_probe")


def _model():
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language="english")
    model.eval()
    return model


def _prefix_module(dec, upto):
    return _range_module(dec, 0, upto)


def _range_module(dec, start, upto):
    import torch.nn.functional as F
    import torch.nn as nn

    class SeanetRange(nn.Module):
        """Runs MimiDecOnly's plan[start:upto+1] on its matching intermediate."""

        def __init__(self):
            super().__init__()
            self.mods = dec.mods
            self.elu = dec.elu
            self.plan = dec.plan[start: upto + 1]

        def forward(self, x):
            stack = []
            for kind, idx, pad in self.plan:
                if kind == "conv":
                    x = self.mods[idx](F.pad(x, (pad, 0)))
                elif kind == "zs":
                    x = self.mods[idx](x)
                elif kind == "elu":
                    x = self.elu(x)
                elif kind == "res_open":
                    stack.append(x)
                elif kind == "res_close":
                    x = x + stack.pop()
            return x

    return SeanetRange()


def _opcheck(path, label):
    import collections
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    return dict(sorted(ops.items(), key=lambda kv: -kv[1]))


def _export_range(torch, B, model, win, start, upto, input_channels, input_len, tag,
                  make_fp16=False):
    dec = B.MimiDecOnly(model, L=win).eval()
    n = len(dec.plan)
    if not 0 <= start <= upto < n:
        raise SystemExit(f"range {start}..{upto} out of range 0..{n - 1}")
    example = torch.zeros(1, input_channels, input_len)
    g = _range_module(dec, start, upto).eval()
    with torch.no_grad():
        ref = g(example)

    import litert_torch
    em = litert_torch.convert(g, (example,))
    out = os.path.join(PROBE, f"seanet_{tag}.tflite")
    if os.path.exists(out):
        os.remove(out)
    em.export(out)
    ops = _opcheck(out, tag)
    meta = {
        "win": win,
        "start": start,
        "upto": upto,
        "plan_len": n,
        "plan": [list(p) for p in dec.plan[start: upto + 1]],
        "input_shape": list(example.shape),
        "ref_shape": list(ref.shape),
        "out": out,
        "ops": ops,
        "size_mb": os.path.getsize(out) / 1e6,
    }
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    kinds = "".join(("c" if k == "conv" else "z" if k == "zs" else
                     "e" if k == "elu" else "(" if k == "res_open" else ")")
                    for k, _, _ in dec.plan[start: upto + 1])
    print(f"EXPORT {tag}: win={win} range={start}..{upto}/{n - 1} "
          f"in={tuple(example.shape)} out={tuple(ref.shape)} "
          f"{os.path.getsize(out)/1e6:.2f}MB")
    print(f"  plan: {kinds}")
    print(f"  ops : {ops}")
    print(f"  file: {out}")
    if make_fp16:
        fp16_out = out.replace(".tflite", "_fp16.tflite")
        B.to_fp16(out, fp16_out)
    return out


def _export_one(torch, B, model, win, upto, tag):
    return _export_range(torch, B, model, win, 0, upto, B.MIMI_D, win, tag)


def _patch_zs(mode):
    """Monkeypatch ZeroStuffConvT1d.forward to isolate the crashing op.

    The layer is `nearest upsample x zero-stuff mask -> conv1d -> crop`.
    Modes keep every step but one so the NPU compiler can be blamed on a
    single TFLite op family (RESIZE_NEAREST_NEIGHBOR vs CONV_2D vs MUL/SLICE).

    Replacement modes (candidate fixes for a crashing resize):
    ``padinsert``  true zero-stuff: reshape (B,C,L,1) + pad last dim + reshape
    ``convtr``     the upstream ConvTranspose1d, kernel rebuilt from self.w
    """
    if mode == "orig":
        return
    import torch
    import torch.nn.functional as F
    import build_pockettts as B

    def convtr_weight(self):
        # undo ZeroStuffConvT1d.__init__ (grouped permute + flip) -> ct.weight
        g, k = self.g, self.k
        v = self.w.flip(2)                      # [cout, cin/g, k]
        cout, cin_g = v.shape[0], v.shape[1]
        return (v.view(g, cout // g, cin_g, k).permute(0, 2, 1, 3)
                .reshape(cin_g * g, cout // g, k).contiguous())

    def forward(self, x):
        if mode == "padinsert":
            xn = x.reshape(x.shape[0], x.shape[1], self.L, 1)
            xn = F.pad(xn, (0, self.s - 1))
            xn = xn.reshape(x.shape[0], x.shape[1], self.L * self.s)
        elif mode == "convtr":
            ct_w = convtr_weight(self)
            y = F.conv_transpose1d(x, ct_w, bias=self.b, stride=self.s)
            return y[:, :, :self.L * self.s]
        elif mode == "expand":
            xn = (x.unsqueeze(-1).expand(*x.shape[:-1], self.s)
                  .reshape(x.shape[0], x.shape[1], self.L * self.s)) * self.mask
        elif mode == "nomask":
            xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * self.s),
                               mode="nearest").squeeze(2)
        elif mode == "padonly":
            xn = F.pad(x, (0, self.L * (self.s - 1))) * self.mask
        else:
            xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * self.s),
                               mode="nearest").squeeze(2) * self.mask
        y = F.conv1d(xn, self.w, bias=self.b, padding=self.k - 1, groups=self.g)
        if mode == "noconv":
            return xn
        if mode == "noslice":
            return y
        return y[:, :, :self.L * self.s]

    B.ZeroStuffConvT1d.forward = forward


def do_zs(args):
    """Export just the first ZeroStuffConvT1d (decoder mod 1) for isolation."""
    import torch
    import build_pockettts as B

    os.makedirs(PROBE, exist_ok=True)
    win = args.win if args.win else B.S_DEC
    _patch_zs(args.zs_mode)
    model = _model()
    dec = B.MimiDecOnly(model, L=win).eval()
    zs = dec.mods[1]

    class ZsOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zs = zs

        def forward(self, x):
            return self.zs(x)

    g = ZsOnly().eval()
    with torch.no_grad():
        ref = g(torch.zeros(1, B.MIMI_D, win))
    import litert_torch
    em = litert_torch.convert(g, (torch.zeros(1, B.MIMI_D, win),))
    tag = args.tag or f"zsonly_w{win}_{args.zs_mode}"
    out = os.path.join(PROBE, f"seanet_{tag}.tflite")
    if os.path.exists(out):
        os.remove(out)
    em.export(out)
    ops = _opcheck(out, tag)
    print(f"EXPORT {tag}: in=(1,{B.MIMI_D},{win}) out={tuple(ref.shape)} "
          f"mode={args.zs_mode} {os.path.getsize(out)/1e6:.2f}MB")
    print(f"  ops : {ops}")
    print(f"  file: {out}")
    return out


def do_export(args):
    import torch
    import build_pockettts as B

    os.makedirs(PROBE, exist_ok=True)
    win = args.win if args.win else B.S_DEC
    _patch_zs(args.zs_mode)
    model = _model()
    upto = args.upto if args.upto is not None else len(B.MimiDecOnly(model, L=win).plan) - 1
    tag = args.tag or f"w{win}_u{upto:02d}"
    if args.zs_mode != "orig":
        tag += "_" + args.zs_mode
    return _export_one(torch, B, model, win, upto, tag)


def do_range(args):
    """Export a plan range whose input is an intermediate decoder tensor."""
    import torch
    import build_pockettts as B

    os.makedirs(PROBE, exist_ok=True)
    win = args.win if args.win else B.S_DEC
    _patch_zs(args.zs_mode)
    model = _model()
    tag = args.tag or f"range_w{win}_{args.start:02d}-{args.upto:02d}_{args.zs_mode}"
    _export_range(torch, B, model, win, args.start, args.upto,
                  args.input_channels, args.input_len, tag, args.fp16)
    return 0


def _parity(actual, expected):
    import numpy as np

    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr), float(np.max(np.abs(a - b)))


def do_verify(args):
    """Check the near/far composition against padded and original decoders."""
    import numpy as np
    import torch
    import build_pockettts as B

    source = args.input or os.path.join(PROBE, "seanet_input.npy")
    x = np.load(source).astype(np.float32)
    if x.shape != (1, B.MIMI_D, args.win):
        raise SystemExit(f"expected input shape (1,{B.MIMI_D},{args.win}), got {x.shape}")

    model = _model()
    original = B.MimiDecOnly(model, L=args.win).eval()
    xt = torch.from_numpy(x)
    with torch.no_grad():
        original_out = original(xt).cpu().numpy()

    _patch_zs("padinsert")
    dec = B.MimiDecOnly(model, L=args.win).eval()
    near = _range_module(dec, 0, args.cut).eval()
    far = _range_module(dec, args.cut + 1, len(dec.plan) - 1).eval()
    with torch.no_grad():
        intermediate = near(xt)
        composed = far(intermediate)
        padded_full = dec(xt)

    print(f"VERIFY input={tuple(x.shape)} near=0..{args.cut} "
          f"intermediate={tuple(intermediate.shape)} far={args.cut + 1}..{len(dec.plan)-1}")
    for label, value in (("composed_vs_padinsert_full", composed),
                          ("padinsert_full_vs_original", padded_full)):
        expected = padded_full if label.startswith("composed") else original_out
        corr, maxd = _parity(value.cpu().numpy(), expected)
        print(f"  {label}: corr={corr:.9f} max|d|={maxd:.6g}")
    if args.save:
        np.save(args.save, composed.cpu().numpy())


def do_compare(args):
    """Compare device output dumps from SeanetNpuProbeTest."""
    import numpy as np

    def read_bin(name):
        path = os.path.join(args.device_dir, name)
        if not os.path.isfile(path):
            print(f"  {name}: missing")
            return None
        return np.fromfile(path, dtype="<f4")

    def report(label, actual, expected):
        if actual is None or expected is None:
            return
        if actual.size != expected.size:
            print(f"  {label}: size mismatch {actual.size} vs {expected.size}")
            return
        a = actual.astype(np.float64, copy=False)
        b = expected.astype(np.float64, copy=False)
        corr = (np.corrcoef(a, b)[0, 1]
                if np.std(a) > 0 and np.std(b) > 0 else float("nan"))
        maxd = np.max(np.abs(a - b))
        rel = 20.0 * np.log10(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        corr_text = f"{corr:.9f}" if np.isfinite(corr) else "undefined (constant tensor)"
        print(f"  {label}: corr={corr_text} max|d|={maxd:.6g} relDb={rel:.3f}")

    outputs = {name: read_bin(name) for name in (
        "seanet_simple_near_gpu.bin", "seanet_simple_near_npu.bin",
        "seanet_near_gpu.bin", "seanet_near_npu.bin",
        "seanet_near_npu_half.bin", "seanet_near_npu_no_truncation.bin",
        "seanet_far_gpu_same_near_gpu_input.bin",
        "seanet_far_npu_same_near_gpu_input.bin",
        "seanet_far_npu_half_same_input.bin",
        "seanet_far_npu_no_truncation_same_input.bin",
        "seanet_near_int8_gpu.bin", "seanet_near_int8_npu.bin",
        "seanet_far_int8_gpu_same_input.bin", "seanet_far_int8_npu_same_input.bin",
        "seanet_pipeline_gpu.bin", "seanet_pipeline_npu.bin",
        "seanet_pipeline_npu_gpu.bin", "seanet_pipeline_gpu_npu.bin",
        "seanet_pipeline_int8_gpu.bin", "seanet_pipeline_int8_npu.bin",
        "seanet_pipeline_int8_npu_gpu.bin",
        "seanet_full_gpu_gold.bin",
    )}
    ref_path = args.torch_ref or os.path.join(PROBE, "seanet_ref.npy")
    torch_ref = np.load(ref_path).astype(np.float32).reshape(-1)
    print("Device parity (far GPU/NPU inputs are intentionally the same near-GPU output):")
    report("simple near NPU vs GPU", outputs["seanet_simple_near_npu.bin"],
           outputs["seanet_simple_near_gpu.bin"])
    report("near NPU vs GPU", outputs["seanet_near_npu.bin"],
           outputs["seanet_near_gpu.bin"])
    report("far NPU vs GPU, shape-matched", outputs["seanet_far_npu_same_near_gpu_input.bin"],
           outputs["seanet_far_gpu_same_near_gpu_input.bin"])
    report("int8 near NPU vs GPU", outputs["seanet_near_int8_npu.bin"],
           outputs["seanet_near_int8_gpu.bin"])
    report("int8 far NPU vs GPU, shape-matched",
           outputs["seanet_far_int8_npu_same_input.bin"],
           outputs["seanet_far_int8_gpu_same_input.bin"])
    report("composed NPU vs GPU", outputs["seanet_pipeline_npu.bin"],
           outputs["seanet_pipeline_gpu.bin"])
    report("composed NPU->GPU vs torch", outputs["seanet_pipeline_npu_gpu.bin"], torch_ref)
    report("composed NPU->GPU vs GPU/GPU", outputs["seanet_pipeline_npu_gpu.bin"],
           outputs["seanet_pipeline_gpu.bin"])
    report("composed GPU->NPU vs torch", outputs["seanet_pipeline_gpu_npu.bin"], torch_ref)
    report("int8 composed NPU vs torch", outputs["seanet_pipeline_int8_npu.bin"], torch_ref)
    report("int8 composed NPU->GPU vs torch",
           outputs["seanet_pipeline_int8_npu_gpu.bin"], torch_ref)
    report("int8 composed GPU vs torch", outputs["seanet_pipeline_int8_gpu.bin"], torch_ref)
    report("composed GPU vs torch", outputs["seanet_pipeline_gpu.bin"], torch_ref)
    report("composed NPU vs torch", outputs["seanet_pipeline_npu.bin"], torch_ref)
    report("full GPU gold vs torch", outputs["seanet_full_gpu_gold.bin"], torch_ref)
    if args.torch_stages:
        import torch
        import build_pockettts as B

        input_path = os.path.join(PROBE, "seanet_input.npy")
        x = torch.from_numpy(np.load(input_path).astype(np.float32))
        _patch_zs("padinsert")
        dec = B.MimiDecOnly(_model(), L=x.shape[-1]).eval()
        near = _range_module(dec, 0, 2).eval()
        far = _range_module(dec, 3, len(dec.plan) - 1).eval()
        gpu_near = outputs["seanet_near_gpu.bin"]
        if gpu_near is None:
            return
        with torch.no_grad():
            near_ref = near(x).cpu().numpy().reshape(-1)
            far_ref = far(torch.from_numpy(gpu_near.reshape(1, 256, 3072))).cpu().numpy().reshape(-1)
        print("Device stages vs torch fp32 (far reference is evaluated on the exact near-GPU input):")
        report("near GPU vs torch", outputs["seanet_near_gpu.bin"], near_ref)
        report("near NPU vs torch", outputs["seanet_near_npu.bin"], near_ref)
        report("near NPU half vs torch", outputs["seanet_near_npu_half.bin"], near_ref)
        report("near NPU no_truncation vs torch",
               outputs["seanet_near_npu_no_truncation.bin"], near_ref)
        report("int8 near GPU vs torch", outputs["seanet_near_int8_gpu.bin"], near_ref)
        report("int8 near NPU vs torch", outputs["seanet_near_int8_npu.bin"], near_ref)
        report("far GPU vs torch, same input", outputs["seanet_far_gpu_same_near_gpu_input.bin"],
               far_ref)
        report("far NPU vs torch, same input", outputs["seanet_far_npu_same_near_gpu_input.bin"],
               far_ref)
        report("far NPU half vs torch, same input",
               outputs["seanet_far_npu_half_same_input.bin"], far_ref)
        report("far NPU no_truncation vs torch, same input",
               outputs["seanet_far_npu_no_truncation_same_input.bin"], far_ref)
        int8_gpu_near = outputs["seanet_near_int8_gpu.bin"]
        if int8_gpu_near is not None:
            with torch.no_grad():
                int8_far_ref = far(torch.from_numpy(
                    int8_gpu_near.reshape(1, 256, 3072))).cpu().numpy().reshape(-1)
            report("int8 far GPU vs torch, same input",
                   outputs["seanet_far_int8_gpu_same_input.bin"], int8_far_ref)
            report("int8 far NPU vs torch, same input",
                   outputs["seanet_far_int8_npu_same_input.bin"], int8_far_ref)


def do_sweep(args):
    """Export many prefixes in one process (model load is the slow part)."""
    import torch
    import build_pockettts as B

    os.makedirs(PROBE, exist_ok=True)
    win = args.win if args.win else B.S_DEC
    _patch_zs(args.zs_mode)
    model = _model()
    n = len(B.MimiDecOnly(model, L=win).plan)
    lo = 0 if args.start is None else args.start
    hi = n - 1 if args.end is None else args.end
    for upto in range(lo, hi + 1):
        _export_one(torch, B, model, win, upto,
                    args.tag_prefix + f"w{win}_u{upto:02d}")
    return 0



def do_aot(args):
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.vendors.google_tensor import target as gt

    src = args.file
    if not os.path.isabs(src):
        src = os.path.join(PROBE, src)
    if not os.path.exists(src):
        raise SystemExit(f"no such file: {src}")
    tag = os.path.splitext(os.path.basename(src))[0]
    soc = getattr(gt.SocModel, args.soc)
    target = gt.Target(soc)
    work = tempfile.mkdtemp(prefix="npuseanet_")
    try:
        kwargs = {}
        if args.truncation != "auto":
            kwargs["google_tensor_truncation_type"] = args.truncation
        result = aot_lib.aot_compile(src, output_dir=work, target=[target],
                                     keep_going=False, **kwargs)
        report = result.compilation_report().strip().replace("\n", " | ")
        result.export(work, model_name="m")
        import glob
        produced = glob.glob(os.path.join(work, "m_*.tflite"))
        if not produced:
            raise RuntimeError(f"no compiled model in {work}: {os.listdir(work)}")
        produced = produced[0]
        suffix = "_aot" if args.truncation == "auto" else f"_aot_{args.truncation}"
        dst = os.path.join(PROBE, f"{tag}{suffix}.tflite") if args.out is None else args.out
        shutil.copy(produced, dst)
        print(f"OK   {tag}: {report} -> {os.path.getsize(dst)/1e6:.1f}MB {dst}")
        return 0
    except Exception as e:  # noqa: BLE001 - a crash here is the result
        print(f"FAIL {tag}: {type(e).__name__}: {str(e).splitlines()[0]}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def do_quantize(args):
    """Create an 8-bit weight + activation TFLite graph with range calibration."""
    import numpy as np
    import torch
    import build_pockettts as B

    os.makedirs(PROBE, exist_ok=True)
    tag = args.tag or ("0-2_padinsert" if args.range == "near" else "3-26_padinsert")
    source = os.path.join(PROBE, f"seanet_{args.range}_w512_{tag}.tflite")
    if args.source:
        source = args.source if os.path.isabs(args.source) else os.path.join(PROBE, args.source)
    if not os.path.isfile(source):
        raise SystemExit(f"no such source graph: {source}")

    _patch_zs("padinsert")
    model = _model()
    dec = B.MimiDecOnly(model, L=512).eval()
    near = _range_module(dec, 0, 2).eval()
    range_module = (near if args.range == "near"
                    else _range_module(dec, 3, len(dec.plan) - 1).eval())
    input_path = os.path.join(PROBE, "seanet_input.npy")
    seed_input = np.load(input_path).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    inputs = [seed_input]
    inputs.extend(rng.normal(0.0, 0.5, (1, B.MIMI_D, 512)).astype(np.float32)
                  for _ in range(args.samples - 1))
    if args.range == "far":
        with torch.no_grad():
            inputs = [near(torch.from_numpy(value)).cpu().numpy() for value in inputs]

    calibration = {"serving_default": [{"args_0": value} for value in inputs]}
    spec = {"kind": "static", "ops": ["ALL_SUPPORTED"], "weight_bits": 8,
            "act_bits": 8, "regex": ".*"}
    dst = os.path.join(PROBE, f"seanet_{args.range}_w512_{tag}_int8.tflite")
    B.to_quant(source, dst, spec, calibration=calibration)

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=dst)
    interpreter.allocate_tensors()
    for kind, details in (("input", interpreter.get_input_details()),
                          ("output", interpreter.get_output_details())):
        for detail in details:
            qp = detail["quantization_parameters"]
            print(f"{kind}: name={detail['name']} shape={detail['shape'].tolist()} "
                  f"dtype={detail['dtype'].__name__} scale={qp['scales'][:4]} "
                  f"zero={qp['zero_points'][:4]}")
    test_input = seed_input
    if args.range == "far":
        with torch.no_grad():
            test_input = near(torch.from_numpy(seed_input)).cpu().numpy()
    input_detail = interpreter.get_input_details()[0]
    if input_detail["dtype"] == np.int8:
        scale, zero = input_detail["quantization"]
        test_data = np.clip(np.round(test_input / scale) + zero, -128, 127).astype(np.int8)
    else:
        test_data = test_input.astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], test_data)
    interpreter.invoke()
    output_detail = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_detail["index"])
    if output.dtype == np.int8:
        scale, zero = output_detail["quantization"]
        output = (output.astype(np.float32) - zero) * scale
    with torch.no_grad():
        expected = range_module(torch.from_numpy(test_input)).cpu().numpy()
    corr, maxd = _parity(output, expected)
    rel_db = 20.0 * np.log10(np.linalg.norm(output - expected) /
                             max(np.linalg.norm(expected), 1e-30))
    print(f"  host int8 vs torch: corr={corr:.9f} max|d|={maxd:.6g} relDb={rel_db:.3f}")
    print(f"INT8 {args.range}: source={source} samples={len(inputs)} "
          f"ops={_opcheck(dst, args.range)} -> {dst}")


def do_quant_check(args):
    """Compare calibrated int8 range graphs and their host-composed output."""
    import numpy as np
    import torch
    import build_pockettts as B
    from ai_edge_litert.interpreter import Interpreter

    _patch_zs("padinsert")
    model = _model()
    dec = B.MimiDecOnly(model, L=512).eval()
    near = _range_module(dec, 0, 2).eval()
    far = _range_module(dec, 3, len(dec.plan) - 1).eval()
    x = np.load(os.path.join(PROBE, "seanet_input.npy")).astype(np.float32)

    def run(path, values):
        it = Interpreter(model_path=path)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        scale, zero = inp["quantization"]
        encoded = np.clip(np.round(values / scale) + zero, -128, 127).astype(inp["dtype"])
        it.set_tensor(inp["index"], encoded)
        it.invoke()
        out = it.get_output_details()[0]
        value = it.get_tensor(out["index"])
        if np.issubdtype(value.dtype, np.integer):
            out_scale, out_zero = out["quantization"]
            value = (value.astype(np.float32) - out_zero) * out_scale
        return value.astype(np.float32)

    near_path = os.path.join(PROBE, "seanet_near_w512_0-2_padinsert_int8.tflite")
    far_path = os.path.join(PROBE, "seanet_far_w512_3-26_padinsert_int8.tflite")
    quant_near = run(near_path, x)
    quant_audio = run(far_path, quant_near)
    with torch.no_grad():
        near_ref = near(torch.from_numpy(x)).cpu().numpy()
        audio_ref = dec(torch.from_numpy(x)).cpu().numpy()

    def report(name, actual, expected):
        corr, maxd = _parity(actual, expected)
        rel = 20.0 * np.log10(np.linalg.norm(actual - expected) /
                              max(np.linalg.norm(expected), 1e-30))
        print(f"{name}: corr={corr:.9f} max|d|={maxd:.6g} relDb={rel:.3f}")

    report("near int8 vs torch", quant_near, near_ref)
    report("far int8 on quant-near intermediate vs torch",
           quant_audio, audio_ref)
    print(f"INT8 pipeline: {near_path} -> {far_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="litert_torch export of a plan prefix")
    e.add_argument("upto", type=int, nargs="?", default=None,
                   help="last plan index to include (default: full plan)")
    e.add_argument("--win", type=int, default=None, help="feature window length")
    e.add_argument("--tag", default=None, help="output name tag")
    e.add_argument("--zs-mode", default="orig",
                   choices=["orig", "expand", "nomask", "padonly", "noconv", "noslice",
                            "padinsert", "convtr"])
    e.set_defaults(func=do_export)

    r = sub.add_parser("range", help="export an arbitrary plan range")
    r.add_argument("--start", type=int, required=True, help="first plan index")
    r.add_argument("--upto", type=int, required=True, help="last plan index")
    r.add_argument("--win", type=int, default=None, help="decoder's initial feature window")
    r.add_argument("--input-channels", type=int, default=512)
    r.add_argument("--input-len", type=int, default=None,
                   help="range input length (default: --win)")
    r.add_argument("--tag", default=None, help="output name tag")
    r.add_argument("--fp16", action="store_true", help="also emit a weight-only fp16 sibling")
    r.add_argument("--zs-mode", default="orig",
                   choices=["orig", "expand", "nomask", "padonly", "noconv", "noslice",
                            "padinsert", "convtr"])
    r.set_defaults(func=do_range)

    v = sub.add_parser("verify", help="host torch parity for a near/far cut")
    v.add_argument("--win", type=int, default=512)
    v.add_argument("--cut", type=int, default=2)
    v.add_argument("--input", default=None, help="input .npy (default: probe input)")
    v.add_argument("--save", default=None, help="optional composed output .npy")
    v.set_defaults(func=do_verify)

    c = sub.add_parser("compare", help="compare device .bin output dumps")
    c.add_argument("--device-dir", default=os.path.join(PROBE, "device_outputs"))
    c.add_argument("--torch-ref", default=None)
    c.add_argument("--torch-stages", action="store_true",
                   help="also compare near/far device outputs with torch range references")
    c.set_defaults(func=do_compare)

    s = sub.add_parser("sweep", help="export a range of plan prefixes in one process")
    s.add_argument("--start", type=int, default=None)
    s.add_argument("--end", type=int, default=None)
    s.add_argument("--win", type=int, default=None, help="feature window length")
    s.add_argument("--tag-prefix", default="")
    s.add_argument("--zs-mode", default="orig",
                   choices=["orig", "expand", "nomask", "padonly", "noconv", "noslice",
                            "padinsert", "convtr"])
    s.set_defaults(func=do_sweep)

    z = sub.add_parser("zs", help="export only the first ZeroStuffConvT1d (decoder mod 1)")
    z.add_argument("--win", type=int, default=None, help="feature window length")
    z.add_argument("--tag", default=None)
    z.add_argument("--zs-mode", default="orig",
                   choices=["orig", "expand", "nomask", "padonly", "noconv", "noslice",
                            "padinsert", "convtr"])
    z.set_defaults(func=do_zs)

    a = sub.add_parser("aot", help="AOT-compile one exported file for the NPU")
    a.add_argument("file")
    a.add_argument("--soc", default="TENSOR_G5")
    a.add_argument("--out", default=None)
    a.add_argument("--truncation", default="auto",
                   choices=["auto", "half", "bfloat16", "no_truncation"],
                   help="Google Tensor floating-point op truncation mode")
    a.set_defaults(func=do_aot)

    q = sub.add_parser("quantize", help="calibrate a range graph to int8 weights/activations")
    q.add_argument("--range", required=True, choices=["near", "far"])
    q.add_argument("--tag", default=None, help="source graph tag (defaults to standard cut name)")
    q.add_argument("--source", default=None, help="source fp32 TFLite graph")
    q.add_argument("--samples", type=int, default=8)
    q.add_argument("--seed", type=int, default=2026)
    q.set_defaults(func=do_quantize)

    qc = sub.add_parser("quant-check", help="host parity for calibrated int8 near/far graphs")
    qc.set_defaults(func=do_quant_check)

    args = ap.parse_args()
    if args.cmd == "range" and args.input_len is None:
        args.input_len = args.win if args.win is not None else 4096
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
