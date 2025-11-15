import argparse
from . import submit


def main():
    parser = argparse.ArgumentParser(description="Offline mugrade stub.")
    subparsers = parser.add_subparsers(dest="command")
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("api_key")
    submit_parser.add_argument("assignment")
    submit_parser.add_argument("-k", "--key", required=False)

    args = parser.parse_args()
    if args.command == "submit":
        submit(args.api_key, args.assignment, key=args.key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
