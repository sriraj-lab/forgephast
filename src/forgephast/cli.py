"""Top-level CLI dispatcher for forgephast.

Usage::

    forgephast amplify <subcommand> [options]
    forgephast phenotype <subcommand> [options]
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: forgephast {amplify,phenotype} ...\n"
            "\n"
            "HapStore-native population amplifier and polygenic phenotype simulator\n"
            "\n"
            "subcommands:\n"
            "  amplify     Population amplification commands\n"
            "  phenotype   Phenotype simulation commands\n"
            "\n"
            "Run 'forgephast <subcommand> --help' for details."
        )
        sys.exit(0)

    group = args[0]
    remaining = args[1:]

    if group == "amplify":
        sys.argv = ["forgephast amplify"] + remaining
        from forgephast.amplifier.cli import main as amplify_main

        amplify_main()
    elif group == "phenotype":
        sys.argv = ["forgephast phenotype"] + remaining
        from forgephast.phenotype.cli import main as phenotype_main

        phenotype_main()
    else:
        print(f"forgephast: unknown subcommand '{group}'", file=sys.stderr)
        print("usage: forgephast {amplify,phenotype} ...", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
