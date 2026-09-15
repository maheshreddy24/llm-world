"""Manual inspection run: a Crafter episode with the 12b model, saving exactly
what the VLM sees and what it produces, so you can eyeball it before
committing to a real run.

    python debug_run.py --steps 100

Each run gets its own folder under --out (default runs/debug/), e.g.
runs/debug/run_20260101_120000/, containing:
    log.txt           every step's raw output, parsed action and state,
                       one step after another, separated by a divider line
    frame_001.png ..   the annotated image the model was shown, one per step
    video.mp4          those frames with the log burned in below each one,
                        built automatically at the end of the run
"""
import argparse
from pathlib import Path

import crafter

from src.agent import Agent
from src.constants import ACTION_INDEX
from src.episode import describe_outcome, guess_death_cause, parse_response
from src.observation import annotate_frame, format_observation, read_state
from utils import build_video, new_run_dir, write_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-4-12B-it')
    parser.add_argument('--manual', default='crafter_info.txt')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='runs/debug')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--history', type=int, default=16,
                        help='Past steps kept in context, oldest dropped. Matches eval_crafter.py default.')
    parser.add_argument('--video-fps', type=int, default=3)
    args = parser.parse_args()
    width = len(str(args.steps))

    manual_path = Path(args.manual)
    manual = manual_path.read_text() if manual_path.exists() else ''
    if not manual:
        print(f'NOTE: no manual found at {manual_path}, running with an empty rule book '
              '(fine for plumbing, not for judging behaviour).')

    run_dir = new_run_dir(args.out, prefix='debug')

    class Args:
        model = args.model
        dtype = 'bfloat16'
        device = 'cuda'
        attn = None
        detail_steps = 4
        temperature = 0.7
        max_new_tokens = 256
    agent_args = Args()

    print(f'Loading {args.model} ...')
    agent = Agent(agent_args, manual)
    print(f'Loaded: {agent.n_params / 1e9:.2f}B params')
    (run_dir / 'system_prompt.txt').write_text(agent.system)

    env = crafter.Env(seed=args.seed, length=args.steps, reward=True,
                       size=(args.image_size, args.image_size))
    obs = env.reset()
    seen = {}
    state = read_state(env, seen)
    history, plan = [], ''

    log = (run_dir / 'log.txt').open('w')

    for step in range(1, args.steps + 1):
        image = annotate_frame(obs, args.image_size, draw_labels=True)
        observation = format_observation(state, seen)
        frame_name = f'frame_{step:0{width}d}.png'
        image.save(run_dir / frame_name)

        raw, telemetry = agent.act(image, observation, history, plan, step)
        action, plan, reasoning, how = parse_response(raw, plan)

        before = state
        obs, reward, done, _ = env.step(ACTION_INDEX[action])
        state = read_state(env, seen)
        outcome = describe_outcome(before, state)
        dead = state['vitals'].get('health', 1) <= 0
        death_cause = guess_death_cause(state) if dead else None

        write_step(log, step=step, frame_name=frame_name, action=action,
                   plan=plan, reasoning=reasoning, how=how, outcome=outcome,
                   reward=reward, done=done, achievements=len(state['achievements']),
                   died=death_cause, user_text=telemetry['user_text'])

        history.append({'step': step, 'action': action, 'outcome': outcome, 'plan': plan,
                        'reasoning': reasoning[:400]})
        history = history[-args.history:]

        print(f'[step {step}] action={action} ({how}) | '
              f'achievements {len(state["achievements"])} | outcome: {outcome}')
        if dead:
            print(f'  DIED ({death_cause})')
        if done:
            print('Episode ended.')
            break

    log.close()
    print(f'\nSaved {step} steps to {run_dir}/')
    build_video(run_dir, fps=args.video_fps)


if __name__ == '__main__':
    main()
