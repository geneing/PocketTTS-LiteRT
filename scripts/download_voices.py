#!/usr/bin/env python3
"""Download preset Pocket TTS voice states and repack them for the Android app.

``build_pockettts.py assets`` only writes a hand-picked list of six voices. This
pulls the rest of the English presets that ship in ``kyutai/pocket-tts``
(``languages/english/embeddings/*.safetensors``) and repacks them into the same
``pt_voice_<name>.bin`` layout the app loads, so voices can be added without
re-running the whole conversion.

The upstream state is a flow-LM KV cache; the app wants it as fp16 with the key
dimension de-interleaved, prefixed by an int32 frame count --
:func:`build_pockettts.write_voice_blob` is the one implementation of that
format, shared with the bundled voices.

Licensing: ``kyutai/pocket-tts`` is CC-BY-4.0. Some upstream voices derive from
datasets that are *not* redistributable (Expresso and EARS are CC-BY-NC), which
is why ``build_pockettts.py`` bundles only the permissive six. This script
defaults to the permissive set and refuses the known-restricted names unless
``--include-restricted`` is passed; check ``kyutai/tts-voices`` before shipping
anything else.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Names whose upstream recording is CC-BY-NC or otherwise not redistributable.
# build_pockettts.py's stage_assets excludes these from the bundle for the same
# reason; listing them here keeps the two in step.
RESTRICTED = {"cosette", "jean"}

# Shipped by `build_pockettts.py assets`; offered here for convenience only.
BUNDLED = ["alba", "marius", "javert", "charles", "mary", "eve"]


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "out"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*",
                        help=f"voices to fetch (default: every permissive English preset "
                             f"not in {BUNDLED})")
    parser.add_argument("--all", action="store_true",
                        help="fetch every English preset, restricted names excluded")
    parser.add_argument("--include-restricted", action="store_true",
                        help=f"also fetch {sorted(RESTRICTED)} (non-commercial upstream)")
    parser.add_argument("--list", action="store_true",
                        help="print the available presets and exit")
    parser.add_argument("--repo", default="kyutai/pocket-tts",
                        help="Hugging Face repo holding the embeddings (default: %(default)s)")
    parser.add_argument("--revision", default="main",
                        help="repo revision (default: %(default)s); the embedder is unchanged "
                             "from 001cf6e, so `main` states stay valid")
    parser.add_argument("--language", default="english",
                        help="language subdirectory (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("PT_OUT", default_out)),
                        help=f"output directory (default: PT_OUT or {default_out})")
    parser.add_argument("--force", action="store_true",
                        help="replace voice files that already exist")
    return parser.parse_args()


def list_presets(repo: str, revision: str, language: str) -> list[str]:
    """Sorted English preset names that have an embedding in the repo."""
    from huggingface_hub import list_repo_files
    prefix = f"languages/{language}/embeddings/"
    return sorted(
        Path(f).stem for f in list_repo_files(repo, revision=revision)
        if f.startswith(prefix) and f.endswith(".safetensors")
    )


def main() -> int:
    args = parse_args()

    try:
        from build_pockettts import load_voice_state, write_voice_blob
    except ImportError as exc:
        print(f"error: run this from the scripts/ directory ({exc})", file=sys.stderr)
        return 2

    try:
        available = list_presets(args.repo, args.revision, args.language)
    except Exception as exc:
        print(f"error: could not list {args.repo}: {exc}", file=sys.stderr)
        print("hint: kyutai/pocket-tts is gated; authenticate with `huggingface-cli login` "
              "after accepting its terms.", file=sys.stderr)
        return 1

    if args.list:
        for name in available:
            marks = []
            if name in RESTRICTED:
                marks.append("non-commercial")
            if name in BUNDLED:
                marks.append("bundled")
            print(f"{name}{'  (' + ', '.join(marks) + ')' if marks else ''}")
        return 0

    if args.names:
        wanted = [n.strip().lower() for n in args.names]
    elif args.all:
        wanted = available
    else:
        wanted = [n for n in available if n not in BUNDLED and n not in RESTRICTED]

    unknown = [n for n in wanted if n not in available]
    if unknown:
        print(f"error: no such preset in {args.repo}: {', '.join(unknown)}", file=sys.stderr)
        return 2

    restricted = [n for n in wanted if n in RESTRICTED]
    if restricted and not args.include_restricted:
        print(f"error: {', '.join(restricted)} is non-commercial upstream "
              "(Expresso/EARS); pass --include-restricted to fetch it anyway",
              file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, failed = 0, 0, []

    for name in wanted:
        path = args.output_dir / f"pt_voice_{name}.bin"
        if path.exists() and not args.force:
            print(f"{name}: {path.name} exists, skipped (--force to replace)")
            skipped += 1
            continue
        try:
            ks, vs, off = load_voice_state(name, repo=args.repo,
                                           language=args.language, revision=args.revision)
            write_voice_blob(path, ks, vs, off)
        except Exception as exc:
            print(f"{name}: failed: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        print(f"{name}: T={off} -> {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        written += 1

    print(f"\n{written} written, {skipped} skipped, {len(failed)} failed in {args.output_dir}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    if written:
        print("To install: scripts/install_to_device.sh  (pushes every pt_voice_*.bin)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())