#!/usr/bin/env python3
"""Create a PocketTTS-LiteRT voice cache from a WAV and its transcript.

The cache uses the same packed fp16 KV format as build_pockettts.py, so it can
be placed next to the Android model files as ``pt_voice_<name>.bin``.

Requires the pinned Kyutai pocket-tts reference clone and conversion venv; see
AGENTS.md. Pocket TTS derives its speaker state from audio alone. The transcript
is kept as a companion file and can optionally be used to make a preview.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

LAYERS = 6
HEADS = 16
HEAD_DIM = 64
KV_CAPACITY = 512
BUILTIN_VOICES = {"alba", "marius", "javert", "charles", "mary", "eve"}
NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")

# Extra caches live one level below the model files; mirrors PocketTts.VOICES_DIR
# and download_voices.py.
VOICES_DIR = "voices"


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "out"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="clean speech sample in WAV format")
    parser.add_argument("txt", type=Path, help="UTF-8 transcript of the speech sample")
    parser.add_argument("name", help="voice name (lowercase letters, digits, _ or -)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("PT_OUT", default_out)),
        help=f"model output directory; the voice goes in its {VOICES_DIR}/ "
             f"subdirectory (default: PT_OUT or {default_out})",
    )
    parser.add_argument("--preview", action="store_true", help="also synthesize the transcript to a preview WAV")
    parser.add_argument("--force", action="store_true", help="replace output files if they already exist")
    return parser.parse_args()


def voice_layers(state: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    """Read the upstream TTSModel audio-prompt state and validate its geometry."""
    found: dict[int, dict] = {}
    for module_name, module_state in state.items():
        match = re.fullmatch(r"transformer\.layers\.(\d+)\.self_attn", module_name)
        if match:
            found[int(match.group(1))] = module_state

    if sorted(found) != list(range(LAYERS)):
        raise RuntimeError(
            "unexpected Pocket TTS transformer state; expected six "
            f"transformer.layers.*.self_attn caches, found {sorted(found)}"
        )

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    length: int | None = None
    for layer in range(LAYERS):
        layer_state = found[layer]
        cache = layer_state.get("cache")
        offset = layer_state.get("offset")
        if cache is None or offset is None or cache.ndim != 5 or cache.shape[0] != 2:
            raise RuntimeError(f"invalid KV cache in transformer layer {layer}")
        used = int(offset.reshape(-1)[0].item())
        if used <= 0 or used > cache.shape[2] or used > KV_CAPACITY:
            raise ValueError(
                f"voice uses {used} frames; expected 1..{KV_CAPACITY}. "
                "Use a shorter speech sample."
            )
        if cache.shape[1] != 1 or cache.shape[3:] != (HEADS, HEAD_DIM):
            raise RuntimeError(f"unsupported KV shape in transformer layer {layer}: {tuple(cache.shape)}")
        if length is not None and used != length:
            raise RuntimeError("Pocket TTS layers returned inconsistent cache lengths")
        length = used
        keys.append(cache[0, 0, :used])
        values.append(cache[1, 0, :used])

    assert length is not None
    return keys, values, length


def write_voice(path: Path, keys: list[torch.Tensor], values: list[torch.Tensor], length: int) -> None:
    """Write int32 length followed by de-interleaved fp16 K and fp16 V."""
    perm = torch.cat((torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)))
    packed_k = np.stack([
        key.index_select(-1, perm).permute(1, 0, 2).to(torch.float16).cpu().contiguous().numpy()
        for key in keys
    ])
    packed_v = np.stack([
        value.permute(1, 0, 2).to(torch.float16).cpu().contiguous().numpy()
        for value in values
    ])

    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as output:
        np.asarray([length], dtype="<i4").tofile(output)
        packed_k.astype("<f2", copy=False).tofile(output)
        packed_v.astype("<f2", copy=False).tofile(output)
        output.flush()
        os.fsync(output.fileno())
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    name = args.name.strip().lower()
    if not NAME_RE.fullmatch(name):
        print("error: name must match [a-z][a-z0-9_-]{0,63}", file=sys.stderr)
        return 2
    if not args.wav.is_file():
        print(f"error: WAV not found: {args.wav}", file=sys.stderr)
        return 2
    if not args.txt.is_file():
        print(f"error: transcript not found: {args.txt}", file=sys.stderr)
        return 2
    try:
        transcript = args.txt.read_text(encoding="utf-8-sig").strip()
    except UnicodeError as exc:
        print(f"error: transcript must be UTF-8: {exc}", file=sys.stderr)
        return 2
    if not transcript:
        print("error: transcript is empty", file=sys.stderr)
        return 2

    voices_dir = args.output_dir / VOICES_DIR
    voice_path = voices_dir / f"pt_voice_{name}.bin"
    transcript_path = voices_dir / f"pt_voice_{name}.txt"
    preview_path = voices_dir / f"pt_voice_{name}_preview.wav"
    destinations = [voice_path, transcript_path] + ([preview_path] if args.preview else [])
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        print(
            "error: output already exists (pass --force to replace): "
            + ", ".join(map(str, existing)),
            file=sys.stderr,
        )
        return 2
    if name in BUILTIN_VOICES:
        print(f"warning: {name!r} is a bundled voice name; this will replace its cache")

    try:
        from pocket_tts import TTSModel

        print("Loading Pocket TTS English model...")
        model = TTSModel.load_model(language="english")
        model.eval()
        if not model.has_voice_cloning:
            raise RuntimeError(
                "the loaded weights do not support voice cloning. Accept the terms for "
                "kyutai/pocket-tts on Hugging Face, authenticate, and retry."
            )
        print(f"Encoding voice sample: {args.wav}")
        state = model.get_state_for_audio_prompt(str(args.wav))
        keys, values, length = voice_layers(state)

        voices_dir.mkdir(parents=True, exist_ok=True)
        write_voice(voice_path, keys, values, length)
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        print(f"Created {VOICES_DIR}/{voice_path.name} ({length} frames)")
        print(f"Saved transcript {VOICES_DIR}/{transcript_path.name}")

        if args.preview:
            from scipy.io import wavfile

            print("Synthesizing transcript preview...")
            audio = model.generate_audio(state, transcript)
            wavfile.write(
                str(preview_path),
                model.sample_rate,
                audio.detach().cpu().numpy().squeeze().astype(np.float32),
            )
            print(f"Created {preview_path}")
    except Exception as exc:
        # Do not leave a seemingly valid cache if an error occurs after writing.
        if voice_path.exists() and not transcript_path.exists():
            voice_path.unlink()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("The transcript does not affect the embedding; Pocket TTS conditions on the WAV.")
    print("To install, push the .bin into the app's external files directory and restart TTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
