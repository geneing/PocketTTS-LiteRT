#!/usr/bin/env python3
"""Standalone Tensor_G5 AOT reproducer for large nearest-neighbour resize.

The export and AOT environments may be separate. Use ``export`` with the
litert-torch environment, then ``aot`` or ``sweep`` with the LiteRT SDK
environment. The default graph is only a nearest-neighbour upsample from
[1, 512, 4096] to [1, 512, 24576]; optional flags add the SEANet mask and
grouped Conv1D path for further isolation.
"""

import argparse
import os
import re
import tempfile
import traceback


def make_module(length, stride, channels, mask, conv, groups):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResizeRepro(nn.Module):
        def __init__(self):
            super().__init__()
            self.length = length
            self.stride = stride
            self.register_buffer("mask", torch.arange(length * stride)[None, None]
                                 .remainder(stride).eq(0).to(torch.float32))
            if conv:
                if channels % groups:
                    raise ValueError("channels must be divisible by groups")
                self.weight = nn.Parameter(torch.ones(channels, channels // groups, 12),
                                           requires_grad=False)
                self.bias = nn.Parameter(torch.zeros(channels), requires_grad=False)
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

        def forward(self, x):
            # Preserve the 2-D resize lowering used by the original model.
            y = F.interpolate(x.unsqueeze(2), size=(1, self.length * self.stride),
                              mode="nearest").squeeze(2)
            if mask:
                y = y * self.mask
            if conv:
                y = F.conv1d(y, self.weight, self.bias, padding=11, groups=groups)
                y = y[:, :, :self.length * self.stride]
            return y

    return ResizeRepro().eval()


def export_model(args):
    import torch
    import litert_torch

    module = make_module(args.length, args.stride, args.channels,
                         args.mask, args.conv, args.groups)
    example = torch.zeros(1, args.channels, args.length, dtype=torch.float32)
    with torch.no_grad():
        expected = module(example)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    litert_torch.convert(module, (example,)).export(args.output)
    print(f"EXPORTED {args.output}: input={tuple(example.shape)} "
          f"output={tuple(expected.shape)} bytes={os.path.getsize(args.output)}")


def compile_model(model_path):
    from ai_edge_litert.aot import aot_compile as aot_module
    from ai_edge_litert.aot.vendors.google_tensor import target as gt

    work = tempfile.mkdtemp(prefix="npu_crash_repro_")
    print(f"AOT input: {model_path}")
    print(f"AOT work directory (retained for diagnostics): {work}")
    try:
        result = aot_module.aot_compile(model_path, output_dir=work,
                                        target=[gt.Target(gt.SocModel.TENSOR_G5)],
                                        keep_going=False)
        report = result.compilation_report()
        print("OK: Tensor_G5 AOT compilation succeeded")
        print("Compiler report:")
        print(report if report else "<empty>")
        result.export(work, model_name="repro")
        return True
    except Exception as exc:  # The compiler crash is the expected repro outcome.
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print("Compiler report / exception details:")
        print(traceback.format_exc())
        error_match = re.search(r"See (.+?) for details", str(exc))
        diagnostic_files = []
        if error_match and os.path.isfile(error_match.group(1)):
            diagnostic_files.append(error_match.group(1))
        for root, _, files in os.walk(work):
            for name in files:
                if name.endswith((".error", ".log")):
                    diagnostic_files.append(os.path.join(root, name))
        for path in dict.fromkeys(diagnostic_files):
            print(f"--- {path} ---")
            try:
                with open(path, "r", errors="replace") as stream:
                    print(stream.read())
            except OSError as read_error:
                print(f"<could not read: {read_error}>")
        return False


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export a tiny resize graph with litert-torch")
    export.add_argument("--length", type=int, default=4096, help="input length L")
    export.add_argument("--stride", type=int, default=6, help="resize factor s")
    export.add_argument("--channels", type=int, default=512)
    export.add_argument("--mask", action="store_true", help="multiply by SEANet zero-stuff mask")
    export.add_argument("--conv", action="store_true", help="append grouped Conv1D and crop")
    export.add_argument("--groups", type=int, default=1)
    export.add_argument("--output", default=None)
    export.set_defaults(func=export_model)

    aot = sub.add_parser("aot", help="AOT compile one exported graph for Tensor_G5")
    aot.add_argument("model")
    aot.set_defaults(func=lambda args: 0 if compile_model(args.model) else 1)

    sweep = sub.add_parser("sweep", help="AOT compile exported L-specific graphs")
    sweep.add_argument("--lengths", nargs="+", type=int,
                       default=[128, 256, 512, 1024, 2048, 4096])
    sweep.add_argument("--stride", type=int, default=6)
    sweep.add_argument("--model-dir", default=".")
    sweep.add_argument("--name", default="nearest", help="export filename prefix")
    sweep.set_defaults(func=None)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "export" and args.output is None:
        args.output = f"nearest_L{args.length}_s{args.stride}.tflite"
    if args.command == "sweep":
        results = []
        for length in args.lengths:
            model = os.path.join(args.model_dir,
                                 f"{args.name}_L{length}_s{args.stride}.tflite")
            if not os.path.isfile(model):
                print(f"MISSING L={length}: {model} (export this size first)")
                results.append((length, "MISSING"))
                continue
            ok = compile_model(model)
            results.append((length, "OK" if ok else "FAIL"))
        print("\nSweep summary (input [1,512,L], resize output [1,512,L*s]):")
        for length, status in results:
            print(f"L={length}: {status}")
        return 0 if all(status == "OK" for _, status in results) else 1
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
