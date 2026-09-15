"""Evaluate a Gemma 4 VLM playing Crafter from pixels.

Observation encoding and the master prompt live in observation.py. This file is
the run loop, the metrics, and the logging.

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
import math
import re
import time
from collections import Counter
from pathlib import Path

import crafter
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from observation import (
    MASTER_PROMPT, STEP_PROMPT, NO_PLAN, VITALS, FACING_NAMES,
    annotate_frame, format_history, format_observation, read_state)
from utils import build_video, new_run_dir, save_raw_video, write_step

# Gemma 4, released 2026-04-02, Apache 2.0. Re-pull any weights older than the
# 2026-07-15 checkpoint refresh (Flash Attention 4, chat template corrections).
PRESETS = {
    'smoke': 'google/gemma-4-E4B',       # 8B, ~16 GB bf16, fits the 40 GB card
    'mid':   'google/gemma-4-12B-it',    # ~24 GB bf16
    'full':  'google/gemma-4-26b-a4b',   # MoE 26B total / 4B active, ~54 GB bf16, needs 80 GB
    'dense': 'google/gemma-4-31b-it',    # 31B dense, ~62 GB bf16, needs 80 GB
}

# Order must match crafter's action indices; asserted against the env at startup.
ACTIONS = {
    'noop': 'Do nothing for one step. Always applicable.',
    'move_left': 'Walk one tile left. Needs walkable ground there, otherwise you only turn to face left.',
    'move_right': 'Walk one tile right. Needs walkable ground there, otherwise you only turn to face right.',
    'move_up': 'Walk one tile up. Needs walkable ground there, otherwise you only turn to face up.',
    'move_down': 'Walk one tile down. Needs walkable ground there, otherwise you only turn to face down.',
    'do': 'Interact with the tile you are facing: chop a tree for wood, mine stone/coal/iron/diamond, '
          'drink from water, attack a cow/zombie/skeleton, collect a sapling from grass, eat a ripe plant. '
          'Requires the right tool for the material (wood pickaxe for stone, stone pickaxe for iron, '
          'iron pickaxe for diamond).',
    'sleep': 'Close your eyes to restore energy. You are defenceless while asleep, so wall yourself in '
             'first. Waking up safely unlocks an achievement.',
    'place_stone': 'Place a stone on the tile you are facing. Requires stone in inventory.',
    'place_table': 'Place a crafting table on the tile you are facing. Requires wood. '
                   'You must stand next to a table to craft anything.',
    'place_furnace': 'Place a furnace on the tile you are facing. Requires stone. '
                     'Needed together with a table for iron tools.',
    'place_plant': 'Plant a sapling on the tile you are facing. Requires a sapling. '
                   'It grows into a fruit you can later eat with "do".',
    'make_wood_pickaxe': 'Craft a wood pickaxe. Requires a nearby table and wood. Lets you mine stone.',
    'make_stone_pickaxe': 'Craft a stone pickaxe. Requires a nearby table, wood and stone. Lets you mine iron.',
    'make_iron_pickaxe': 'Craft an iron pickaxe. Requires a nearby table AND furnace, plus wood, coal and '
                         'iron. Lets you mine diamond.',
    'make_wood_sword': 'Craft a wood sword. Requires a nearby table and wood. Helps you fight.',
    'make_stone_sword': 'Craft a stone sword. Requires a nearby table, wood and stone.',
    'make_iron_sword': 'Craft an iron sword. Requires a nearby table AND furnace, plus wood, coal and iron.',
}

ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
# Longest-first so 'make_wood_pickaxe' wins over 'do' when scanning free text.
ACTION_RE = re.compile(r'\b(' + '|'.join(sorted(ACTIONS, key=len, reverse=True)) + r')\b')

ACHIEVEMENTS = list(getattr(crafter.constants, 'achievements', [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling',
    'collect_stone', 'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant',
    'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
    'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant', 'place_stone',
    'place_table', 'wake_up']))

ACHIEVEMENT_DEPTH = {
    'collect_wood': 1, 'place_table': 2, 'eat_cow': 1, 'collect_sapling': 1, 'collect_drink': 1,
    'make_wood_pickaxe': 3, 'make_wood_sword': 3, 'place_plant': 2, 'defeat_zombie': 2,
    'collect_stone': 4, 'place_stone': 5, 'eat_plant': 3, 'defeat_skeleton': 4,
    'make_stone_pickaxe': 5, 'make_stone_sword': 5, 'wake_up': 2, 'place_furnace': 6,
    'collect_coal': 5, 'collect_iron': 6, 'make_iron_pickaxe': 7, 'make_iron_sword': 7,
    'collect_diamond': 8,
}


def format_actions():
    return '\n'.join(f'- {name}: {description}' for name, description in ACTIONS.items())


def describe_outcome(before, after):
    """Say what the last action actually changed, so the model can tell it was wasted."""
    parts = []
    if before['pos'] != after['pos']:
        parts.append('you moved one tile')
    elif before['facing'] != after['facing']:
        parts.append(f'you turned to face {FACING_NAMES.get(after["facing"], "?")} without moving')

    old = {**before['vitals'], **before['inventory']}
    new = {**after['vitals'], **after['inventory']}
    for key in sorted(set(old) | set(new)):
        delta = new.get(key, 0) - old.get(key, 0)
        if delta:
            parts.append(f'{key} {delta:+d}')

    unlocked = after['achievements'] - before['achievements']
    if unlocked:
        parts.append('UNLOCKED ' + ', '.join(sorted(unlocked)))
    return '; '.join(parts) if parts else 'nothing changed, the action had no effect'


def parse_response(text, previous_plan):
    """Extract plan, reasoning and action.

    Returns (action, plan, reasoning, how) where how is 'tag' (clean),
    'scan' (recovered from free text) or 'fail' (nothing usable -> noop).
    A missing <plan> inherits the previous plan rather than blanking it; a
    dropped tag should not reset the agent's goal.
    """
    match = re.search(r'<plan>(.*?)(?:</plan>|$)', text, re.S)
    plan = match.group(1).strip() if match else ''
    plan = plan or previous_plan

    match = re.search(r'<reasoning>(.*?)(?:</reasoning>|$)', text, re.S)
    reasoning = match.group(1).strip() if match else text.strip()

    match = re.search(r'<action>(.*?)(?:</action>|$)', text, re.S)
    candidate = match.group(1) if match else ''
    name = re.sub(r'[^a-z_ ]', '', candidate.lower()).strip().replace(' ', '_')
    if name in ACTIONS:
        return name, plan, reasoning, 'tag'

    for haystack in (candidate.lower(), text.lower()):
        hits = ACTION_RE.findall(haystack)
        if hits:
            return hits[-1], plan, reasoning, 'scan'
    return 'noop', plan, reasoning, 'fail'


def crafter_score(success_rates):
    """Official Crafter score: geometric mean of per-achievement success rates (percent)."""
    rates = np.array([success_rates.get(name, 0.0) for name in ACHIEVEMENTS], dtype=float)
    return float(np.exp(np.mean(np.log(1.0 + rates))) - 1.0)


class Agent:
    def __init__(self, args, manual):
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model)
        torch_dtype = getattr(torch, args.dtype)
        kwargs = dict(device_map=args.device)
        if args.attn:
            kwargs['attn_implementation'] = args.attn
        try:  # transformers renamed torch_dtype -> dtype
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch_dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model, torch_dtype=torch_dtype, **kwargs)
        self.model.eval()
        self.model.requires_grad_(False)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.system = MASTER_PROMPT.format(manual=manual, actions=format_actions())
        self.prompt_tokens = 0
        self.output_tokens = 0

    @torch.inference_mode()
    def act(self, image, observation, history, plan, step):
        """One frame, text history. See module docstring for why."""
        user_text = STEP_PROMPT.format(
            step=step, observation=observation,
            history=format_history(history, self.args.detail_steps),
            plan=plan or NO_PLAN)
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                         {'type': 'text', 'text': user_text}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.model.device)
        prompt_length = int(inputs['input_ids'].shape[1])

        started = time.monotonic()
        sample = self.args.temperature > 0
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=sample,
            temperature=self.args.temperature if sample else None,
            top_p=0.95 if sample else None,
            top_k=64 if sample else None)
        latency = time.monotonic() - started

        new_tokens = generated[0][prompt_length:]
        self.prompt_tokens += prompt_length
        self.output_tokens += int(new_tokens.shape[0])
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return text, {'prompt_tokens': prompt_length,
                      'output_tokens': int(new_tokens.shape[0]),
                      'latency': latency,
                      'user_text': user_text}


def run_episode(agent, args, seed, log_file, text_log, record_video, deadline, out):
    env = crafter.Env(seed=seed, length=args.episode_length, reward=True,
                      size=(args.image_size, args.image_size))
    names = list(getattr(env, 'action_names', []))
    if names:
        assert names == list(ACTIONS), f'Action dict must match crafter action order: {names}'

    obs = env.reset()
    seen = {}                       # landmark memory, persists across the episode
    state = read_state(env, seen)
    history, plan = [], ''
    frames = [obs] if record_video else None
    reward_total, done, timed_out, completed = 0.0, False, False, 0
    parse_counts, action_counts = Counter(), Counter()
    started = time.monotonic()

    for step in range(1, args.episode_length + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break

        image = annotate_frame(obs, args.image_size, draw_labels=args.grid_labels)
        observation = format_observation(state, seen)
        frame_name = None
        if args.dump_frames and step % args.dump_frames == 0:
            frame_name = f'frames/seed{seed}_step{step:05d}.png'
            image.save(out / frame_name)

        raw, telemetry = agent.act(image, observation, history, plan, step)
        action, plan, reasoning, how = parse_response(raw, plan)
        parse_counts[how] += 1
        action_counts[action] += 1

        before = state
        obs, reward, done, _ = env.step(ACTION_INDEX[action])
        state = read_state(env, seen)
        outcome = describe_outcome(before, state)
        reward_total += reward
        completed = step
        if record_video and len(frames) < args.video_max_frames:
            frames.append(obs)

        history.append({'step': step, 'action': action, 'outcome': outcome, 'plan': plan,
                        'reasoning': reasoning[:args.reasoning_chars]})
        history = history[-args.history:]

        record = {
            'seed': seed, 'step': step,
            'action': action, 'plan': plan, 'parse': how,
            'reasoning': reasoning, 'raw': raw, 'outcome': outcome,
            'reward': reward, 'cumulative_reward': reward_total, 'done': done,
            'pos': list(state['pos']),
            'facing': FACING_NAMES.get(state['facing'], '?'),
            'faced_tile': state['faced'], 'faced_label': state['faced_label'],
            'vitals': {k: state['vitals'].get(k, 0) for k in VITALS},
            'inventory': state['inventory'],
            'landmarks': {k: list(v) for k, v in seen.items()},
            'unlocked': sorted(state['achievements']),
            'prompt_tokens': telemetry['prompt_tokens'],
            'output_tokens': telemetry['output_tokens'],
            'latency': round(telemetry['latency'], 3),
        }
        if args.log_prompts:
            record['user_text'] = telemetry['user_text']
        log_file.write(json.dumps(record) + '\n')
        log_file.flush()

        write_step(text_log, step=step, frame_name=frame_name, raw=raw, action=action,
                   plan=plan, reasoning=reasoning, how=how, outcome=outcome,
                   reward=reward, done=done,
                   user_text=telemetry['user_text'] if args.log_prompts else None)

        if args.verbose:
            flag = '' if how == 'tag' else f' ({how})'
            print(f'  [{seed}:{step}] {action}{flag} | plan: {plan[:40]} | '
                  f'R {reward_total:.1f} | {outcome}', flush=True)

        if step == args.preflight_steps:
            clean = parse_counts['tag'] / step
            print(f'  preflight: {clean:.0%} clean <action> tags in first {step} steps', flush=True)
            if clean < 0.5:
                print('  WARNING: format adherence is poor. The score below measures parsing, '
                      'not gameplay. Use a larger preset or constrained decoding.', flush=True)
        if done:
            break

    unlocked = sorted(state['achievements'])
    return {
        'seed': seed, 'steps': completed, 'reward': reward_total,
        'parse_counts': dict(parse_counts), 'action_counts': dict(action_counts),
        'achievements': unlocked,
        'depth': max([ACHIEVEMENT_DEPTH.get(a, 0) for a in unlocked], default=0),
        'landmarks_found': sorted(seen),
        'seconds': time.monotonic() - started,
        'complete': bool(done), 'timed_out': timed_out,
        'frames': frames,
    }


def aggregate(episodes, agent, total_steps, elapsed, args):
    n = len(episodes)
    rates = {name: 100.0 * sum(name in ep['achievements'] for ep in episodes) / n
             for name in ACHIEVEMENTS}
    rewards = [ep['reward'] for ep in episodes]
    depths = [ep['depth'] for ep in episodes]
    actions, parses = Counter(), Counter()
    for ep in episodes:
        actions.update(ep['action_counts'])
        parses.update(ep['parse_counts'])
    total_actions = sum(actions.values()) or 1
    return {
        'model': args.model, 'model_params': agent.n_params,
        'history_length': args.history, 'detail_steps': args.detail_steps,
        'grid_labels': args.grid_labels, 'image_size': args.image_size,
        'temperature': args.temperature,
        'episodes': n,
        'episodes_complete': sum(ep['complete'] for ep in episodes),
        'episodes_truncated': sum(not ep['complete'] for ep in episodes),
        'env_steps': total_steps,
        'crafter_score': crafter_score(rates),
        'reward_mean': float(np.mean(rewards)),
        'reward_std_err': float(np.std(rewards) / math.sqrt(n)),
        'achievement_depth_mean': float(np.mean(depths)),
        'achievement_depth_max': int(max(depths)),
        'success_rates': {k: round(v, 2) for k, v in sorted(rates.items(), key=lambda x: -x[1])},
        'parse_clean_rate': parses['tag'] / total_actions,
        'parse_recovered_rate': parses['scan'] / total_actions,
        'parse_failed_rate': parses['fail'] / total_actions,
        'action_distribution': {k: round(v / total_actions, 4) for k, v in actions.most_common()},
        'steps_per_second': total_steps / max(elapsed, 1e-9),
        'seconds_per_step': elapsed / max(total_steps, 1),
        'prompt_tokens': agent.prompt_tokens, 'output_tokens': agent.output_tokens,
        'tokens_per_step': (agent.prompt_tokens + agent.output_tokens) / max(total_steps, 1),
        'wall_clock_hours': elapsed / 3600.0,
    }


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
