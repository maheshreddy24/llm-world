"""The shared human-readable per-step log format (log.txt), written by
eval_crafter.py and debug_run.py and read back by utils/video.py."""
import json
import re
import time
from pathlib import Path

STEP_RE = re.compile(
    r'STEP (\d+)\s+\(frame: (\S+)\).*?'
    r'--- RAW OUTPUT ---\n(.*?)\n\n'
    r'--- PARSED ---\naction: (.*?)\nplan: (.*?)\nreasoning: (.*?)\nhow: (.*?)\n\n'
    r'--- STATE ---\n(.*?)\n\n',
    re.S)


def write_step(log_file, *, step, frame_name, raw, action, plan, reasoning, how,
               outcome, reward, done, user_text=None):
    """One step's block in the shared log.txt format.

    frame_name is None when no frame was saved this step (e.g. eval_crafter.py's
    --dump-frames only saves every Nth step); build_video skips those for video.
    """
    log_file.write(f'{"=" * 80}\nSTEP {step}  (frame: {frame_name or "none"})\n{"=" * 80}\n\n')
    if user_text is not None:
        log_file.write(f'--- USER PROMPT ---\n{user_text}\n\n')
    log_file.write(f'--- RAW OUTPUT ---\n{raw}\n\n')
    log_file.write(f'--- PARSED ---\naction: {action}\nplan: {plan}\nreasoning: {reasoning}\n'
                    f'how: {how}\n\n')
    state_summary = json.dumps({'outcome': outcome, 'reward': reward, 'done': done})
    log_file.write(f'--- STATE ---\n{state_summary}\n\n')
    log_file.flush()


def parse_log(log_path):
    """Inverse of write_step: log.txt -> list of step dicts."""
    text = Path(log_path).read_text()
    steps = []
    for m in STEP_RE.finditer(text):
        step, frame, raw, action, plan, reasoning, how, state_json = m.groups()
        state = json.loads(state_json)
        steps.append({
            'step': int(step), 'frame': None if frame == 'none' else frame,
            'raw': raw, 'action': action.strip(), 'plan': plan.strip(),
            'reasoning': ' '.join(reasoning.split()), 'how': how.strip(),
            'outcome': state['outcome'], 'reward': state['reward'], 'done': state['done'],
        })
    return steps


def new_run_dir(base, prefix='run'):
    """base/prefix_<timestamp>/, so repeated runs never collide or overwrite."""
    run_dir = Path(base) / f'{prefix}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
