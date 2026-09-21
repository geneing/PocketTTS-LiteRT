#!/usr/bin/env python3
"""Pack `scripts/out/` into the model zips described by `models.json`.

Entries are stored uncompressed: .tflite files barely compress, and stored
entries let the runtime extract them without a deflate pass.

  scripts/pack_models.py [--out scripts/out] [--dist scripts/dist] \
      [--manifest models.json] [--version 2026.09]

Rewrites sha256/bytes in --manifest (in place by default) so the manifest and
the zips can be uploaded to a GitHub release together, then:

  gh release create models-2026.09 --title "Models 2026.09" \
      scripts/dist/pockettts-*.zip scripts/dist/models.json
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/out", help="directory with built models")
    ap.add_argument("--dist", default="scripts/dist", help="where zips + models.json are written")
    ap.add_argument("--manifest", default="models.json", help="manifest to read and update")
    ap.add_argument("--version", default=None, help="override the version in file names")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    version = args.version or manifest["modelVersion"]
    os.makedirs(args.dist, exist_ok=True)

    missing = []
    for variant in manifest["variants"]:
        zip_name = variant["zip"]
        if args.version:
            zip_name = zip_name.replace(manifest["modelVersion"], args.version)
            variant["zip"] = zip_name
        path = os.path.join(args.dist, zip_name)
        present = []
        absent = []
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for name in variant["files"]:
                src = os.path.join(args.out, name)
                if os.path.isfile(src):
                    z.write(src, name)
                    present.append(name)
                else:
                    absent.append(name)
        if not present:
            os.remove(path)
            missing.extend(f"{zip_name}: all files missing ({', '.join(absent)})")
            continue
        variant["sha256"] = sha256(path)
        variant["bytes"] = os.path.getsize(path)
        status = f"{len(present)}/{len(variant['files'])} files"
        if absent:
            status += f", missing {', '.join(absent)}"
            missing.append(f"{zip_name}: {', '.join(absent)}")
        print(f"{zip_name:44s} {variant['bytes']/1e6:8.1f} MB  {status}")

    # Model version in the manifest always matches the packed file names.
    manifest["modelVersion"] = version
    with open(os.path.join(args.dist, "models.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nwrote {args.dist}/models.json and {args.manifest}")
    if missing:
        print("\nwarnings:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
