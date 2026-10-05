#!/usr/bin/env python3
"""Create a dynamic-range int8-weight SEANet graph for a CPU experiment.

Run inside the repository's Linux conversion environment, for example:

    python scripts/quantize_seanet.py \
      --input scripts/out/pt_mimi_deconly_w512.tflite \
      --output scripts/out/seanet_cpu_int8/pt_mimi_deconly_w512_dyn8.tflite

Activations and the host input/output protocol stay float32. The Android
long-paragraph benchmark checks actual CPU latency, energy, and audio quality;
this script only exports the model and checks its tensor interface.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="fp32 streaming SEANet TFLite graph")
    parser.add_argument("--output", required=True, help="output dynamic-range int8 graph")
    args = parser.parse_args()

    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.qtyping import QuantGranularity
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    from ai_edge_litert.interpreter import Interpreter

    recipe = recipe_manager.RecipeManager()
    recipe.add_dynamic_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=8,
        granularity=QuantGranularity.CHANNELWISE,
        algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    quantized = quantizer.Quantizer(float_model=args.input)
    quantized.load_quantization_recipe(recipe.get_quantization_recipe())
    quantized.quantize().export_model(args.output, overwrite=True)

    source = Interpreter(model_path=args.input)
    candidate = Interpreter(model_path=args.output)
    source_inputs = source.get_input_details()
    candidate_inputs = candidate.get_input_details()
    source_outputs = source.get_output_details()
    candidate_outputs = candidate.get_output_details()
    assert [(d["shape"].tolist(), d["dtype"]) for d in source_inputs] == [
        (d["shape"].tolist(), d["dtype"]) for d in candidate_inputs
    ], "quantization changed the input interface"
    assert [(d["shape"].tolist(), d["dtype"]) for d in source_outputs] == [
        (d["shape"].tolist(), d["dtype"]) for d in candidate_outputs
    ], "quantization changed the output interface"

    candidate.allocate_tensors()
    weight_tensors = [
        detail
        for detail in candidate.get_tensor_details()
        if getattr(detail["dtype"], "__name__", str(detail["dtype"])) == "int8"
        and len(detail["shape"]) >= 2
    ]
    assert weight_tensors, "no int8 weight tensors were produced"
    print(
        f"input/output interface preserved: "
        f"{source_inputs[0]['dtype'].__name__}{source_inputs[0]['shape'].tolist()} -> "
        f"{source_outputs[0]['dtype'].__name__}{source_outputs[0]['shape'].tolist()}"
    )
    print(f"int8 weight tensors: {len(weight_tensors)}")
    print(f"exported {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
