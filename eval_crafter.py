"""Evaluate a Gemma 4 VLM playing Crafter from pixels.

Observation encoding and the master prompt live in src/observation.py. The
model wrapper is src/agent.py, the run loop and metrics are src/episode.py.
This file is just argument parsing and the top-level checkpoint loop.

    smoke test (~15 min):
        python eval_crafter.py --preset smoke --smoke

    real run (A100-80, bf16, unquantized):
        python eval_crafter.py --preset full --total-steps 100000 --checkpoint-every 10000

One frame per step, history as text. Feeding previous frames is a tested and
rejected design: EmbodiedBench found adding the last two observation images
DECREASED performance, and FindingDory found frozen VLMs degrade as frame count
rises. The frame is perception for now; the text history is memory.

There is no training here. The policy is the prompt, so more steps only tighten
the confidence intervals -- they do not produce a learning curve.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.agent import Agent
from src.constants import PRESETS, format_actions
from src.episode import aggregate, run_episode
from src.observation import MASTER_PROMPT
from utils import build_video, new_run_dir, save_raw_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=sorted(PRESETS), default='smoke')
    parser.add_argument('--model', default=None, help='Overrides --preset.')
    parser.add_argument('--manual', default='crafter_info.txt')
    parser.add_argument('--output', default=None, help='Defaults to runs/<preset>.')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--smoke', action='store_true',
                        help='15 minute plumbing check: short episodes, tight deadline, verbose.')
    parser.add_argument('--total-steps', type=int, default=20000)
    parser.add_argument('--episode-length', type=int, default=10000)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--max-hours', type=float, default=float('inf'))
    parser.add_argument('--preflight-steps', type=int, default=20)

    parser.add_argument('--history', type=int, default=16,
                        help='HYPERPARAMETER: past steps in context. Sweep this.')
    parser.add_argument('--detail-steps', type=int, default=4,
                        help='Of those, how many keep full reasoning. Older ones compress.')
    parser.add_argument('--grid-labels', action='store_true', default=True,
                        help='Burn A1..I7 labels onto the frame (Set-of-Mark).')
    parser.add_argument('--no-grid-labels', dest='grid_labels', action='store_false',
                        help='Ablation: plain frame, text grid unchanged.')
    parser.add_argument('--reasoning-chars', type=int, default=400)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--attn', default=None,
                        help="e.g. 'eager' or 'flash_attention_2'. Leave unset for Gemma 4.")
    parser.add_argument('--log-prompts', action='store_true',
                        help='Log the full user turn each step. Large, but the only way to '
                             'reconstruct exactly what the model saw.')
    parser.add_argument('--dump-frames', type=int, default=0,
                        help='Save the annotated frame every N steps. 0 disables. '
                             'Use ~25 on the smoke run to eyeball the grid overlay.')
    parser.add_argument('--video-max-frames', type=int, default=2000)
    parser.add_argument('--video-fps', type=int, default=20)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    args.model = args.model or PRESETS[args.preset]
    if args.smoke:
        args.total_steps = min(args.total_steps, 400)
        args.episode_length = min(args.episode_length, 150)
        args.checkpoint_every = min(args.checkpoint_every, 150)
        args.max_hours = min(args.max_hours, 0.25)
        args.verbose = True
        args.dump_frames = args.dump_frames or 25
        args.log_prompts = True
    args.output = args.output or f'runs/{args.preset}'

    torch.manual_seed(args.seed)
    out = new_run_dir(args.output)
    (out / 'videos').mkdir(parents=True, exist_ok=True)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    manual_path = Path(args.manual)
    assert manual_path.exists(), (
        f'Rule book not found at {manual_path}. This is the single biggest lever on '
        'performance -- do not run without it.')
    manual = manual_path.read_text()
    (out / 'master_prompt.txt').write_text(
        MASTER_PROMPT.format(manual=manual, actions=format_actions()))
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')

    agent = Agent(args, manual)
    print(f'{args.model}: {agent.n_params / 1e9:.2f}B params, {args.dtype}', flush=True)
    if torch.cuda.is_available():
        print(f'VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)
    print(f'manual {len(manual)} chars | history {args.history} (detail {args.detail_steps}) | '
          f'grid labels {args.grid_labels} | image {args.image_size}px | '
          f'budget {args.total_steps} steps / {args.max_hours:.2f} h', flush=True)

    episodes, total_steps = [], 0
    next_checkpoint = args.checkpoint_every
    record_next = True
    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    step_log = (out / 'steps.jsonl').open('a')
    text_log = (out / 'log.txt').open('a')
    metrics_log = (out / 'metrics.jsonl').open('a')

    def checkpoint(tag):
        metrics = aggregate(episodes, agent, total_steps, time.monotonic() - started, args)
        metrics['tag'] = tag
        metrics_log.write(json.dumps(metrics) + '\n')
        metrics_log.flush()
        (out / f'metrics_{tag}.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(f'  [{tag}] {total_steps} steps | score {metrics["crafter_score"]:.2f} | '
              f'reward {metrics["reward_mean"]:.2f} | depth {metrics["achievement_depth_max"]} | '
              f'clean parse {metrics["parse_clean_rate"]:.1%} | '
              f'{metrics["tokens_per_step"]:.0f} tok/step', flush=True)
        return metrics

    try:
        while total_steps < args.total_steps and time.monotonic() < deadline:
            seed = args.seed + len(episodes)
            result = run_episode(agent, args, seed, step_log, text_log, record_next, deadline, out)
            frames = result.pop('frames')
            if result['steps'] == 0:
                break
            episodes.append(result)
            total_steps += result['steps']
            elapsed = time.monotonic() - started

            if frames:
                path = out / 'videos' / f'step{total_steps:08d}_seed{seed}.mp4'
                save_raw_video(frames, path, fps=args.video_fps)
                print(f'  video -> {path}', flush=True)
                record_next = False

            print(f'[ep {len(episodes)} seed {seed}] {result["steps"]} steps | '
                  f'R {result["reward"]:.1f} | depth {result["depth"]} | '
                  f'{len(result["achievements"])} achievements | '
                  f'{result["seconds"] / max(result["steps"], 1):.2f}s/step'
                  f'{" | TRUNCATED (deadline)" if result["timed_out"] else ""}', flush=True)

            remaining = args.total_steps - total_steps
            if remaining > 0:
                print(f'  {total_steps}/{args.total_steps} steps | '
                      f'ETA {remaining * (elapsed / total_steps) / 3600.0:.1f} h', flush=True)

            if total_steps >= next_checkpoint:
                checkpoint(f'step{total_steps:08d}')
                while next_checkpoint <= total_steps:
                    next_checkpoint += args.checkpoint_every
                record_next = True
    except KeyboardInterrupt:
        print('\nInterrupted, writing final metrics.', flush=True)
    finally:
        step_log.close()
        text_log.close()
        if episodes:
            print(json.dumps(checkpoint('final'), indent=2))
        else:
            print('No episodes completed.', flush=True)
        metrics_log.close()
        if args.dump_frames:
            build_video(out, fps=args.video_fps)


if __name__ == '__main__':
    main()
