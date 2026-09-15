"""One episode's run loop, response parsing, and the metrics aggregated
across episodes (the official Crafter score and friends)."""
import json
import math
import re
import time
from collections import Counter

import crafter
import numpy as np

from src.constants import ACHIEVEMENT_DEPTH, ACHIEVEMENTS, ACTION_INDEX, ACTION_RE, ACTIONS
from src.observation import FACING_NAMES, VITALS, annotate_frame, format_observation, read_state
from utils import write_step


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


def guess_death_cause(state):
    """Best-effort explanation for a death. Crafter doesn't record a cause
    directly, so this is inferred from what's on screen and vitals at the
    fatal step -- 'unknown' when nothing obvious explains it."""
    window = state['window'].values()
    if state['faced'] == 'lava' or 'lava' in window:
        return 'lava'
    if 'zombie' in window:
        return 'zombie'
    if 'skeleton' in window or 'arrow' in window:
        return 'skeleton'
    vitals = state['vitals']
    if vitals.get('food', 1) <= 0:
        return 'starvation'
    if vitals.get('drink', 1) <= 0:
        return 'dehydration'
    if vitals.get('energy', 1) <= 0:
        return 'exhaustion'
    return 'unknown'


def crafter_score(success_rates):
    """Official Crafter score: geometric mean of per-achievement success rates (percent)."""
    rates = np.array([success_rates.get(name, 0.0) for name in ACHIEVEMENTS], dtype=float)
    return float(np.exp(np.mean(np.log(1.0 + rates))) - 1.0)


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
    reward_total, done, timed_out, completed, dead = 0.0, False, False, 0, False
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
        dead = state['vitals'].get('health', 1) <= 0
        death_cause = guess_death_cause(state) if dead else None
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

        write_step(text_log, step=step, frame_name=frame_name, action=action,
                   plan=plan, reasoning=reasoning, how=how, outcome=outcome,
                   reward=reward, done=done, achievements=len(state['achievements']),
                   died=death_cause,
                   user_text=telemetry['user_text'] if args.log_prompts else None)

        if args.verbose:
            flag = '' if how == 'tag' else f' ({how})'
            print(f'  [{seed}:{step}] {action}{flag} | plan: {plan[:40]} | '
                  f'R {reward_total:.1f} | achievements {len(state["achievements"])} | '
                  f'{outcome}', flush=True)
        if dead:
            print(f'  [{seed}:{step}] DIED ({death_cause}) | '
                  f'{len(state["achievements"])} achievements', flush=True)

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
        'complete': bool(done), 'timed_out': timed_out, 'died': dead,
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
