"""Tiny static file server for scripts/dist, for on-device ReleaseModelSource tests.

  python3 scripts/serve_models.py [--port 8099] [--dir scripts/dist]

Then `adb reverse tcp:8099 tcp:8099` and start ModelDownloadTestActivity.
"""
import argparse
import http.server
import functools
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--dir", default="scripts/dist")
    args = ap.parse_args()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.dir)
    print(f"serving {os.path.abspath(args.dir)} on 0.0.0.0:{args.port}")
    http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
