"""Rebuild the annotated video for a run that already has a log.txt and
frame_*.png files (written by debug_run.py or eval_crafter.py). Both of those
already do this automatically at the end of a run -- use this to regenerate
one without rerunning the model.

    python make_video.py --dir runs/debug/run_20260101_120000
"""
import argparse

from utils import build_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Run folder containing log.txt.')
    parser.add_argument('--fps', type=int, default=3,
                        help='Low by default so each step is readable; bump it up for longer runs.')
    args = parser.parse_args()
    build_video(args.dir, fps=args.fps)


if __name__ == '__main__':
    main()
