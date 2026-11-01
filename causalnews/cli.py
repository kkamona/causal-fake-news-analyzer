"""Command-line interface: analyze, train, evaluate, serve."""

from __future__ import annotations

import argparse
import json
import sys

from .model import cross_validate, default_model_path, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalnews", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse a text")
    source = analyze.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="text to analyse")
    source.add_argument("--file", help="path of a UTF-8 text file")

    sub.add_parser("train", help="train the model and save it")
    sub.add_parser("evaluate", help="5-fold cross-validation on the sample corpus")

    serve = sub.add_parser("serve", help="start the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        from .analyzer import CausalNewsAnalyzer

        text = args.text
        if args.file:
            with open(args.file, encoding="utf-8") as handle:
                text = handle.read()
        print(json.dumps(CausalNewsAnalyzer().analyze(text), indent=2))
    elif args.command == "train":
        path = default_model_path()
        train(path)
        print(f"Model saved to {path}")
    elif args.command == "evaluate":
        print(json.dumps(cross_validate(), indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("causalnews.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
