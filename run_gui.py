"""Launch the interactive Paint Mesh Editing (PME) Studio GUI."""

import argparse
from gui.app import launch_gui


def main():
    parser = argparse.ArgumentParser(description="Paint Mesh Editing (PME) Interactive Studio GUI")
    parser.add_argument(
        '--sample',
        choices=['cow', 'woman'],
        default=None,
        help='Optional sample preset to load immediately on startup.',
    )
    args = parser.parse_args()
    launch_gui(initial_sample=args.sample)


if __name__ == '__main__':
    main()
