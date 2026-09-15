"""The shared human-readable per-step log format (log.txt), written by
eval_crafter.py and debug_run.py and read back by utils/video.py."""
import json
import re
import time
from pathlib import Path

STEP_RE = re.compile(
    r'STEP (\d+)\s+\(frame: (\S+)\).*?'
    r'plan: (.*?)\n'
    r'reasoning: (.*?)\n'
    r'outcome: (.*?)\n'
    r'achievements: (\d+)\n'
    r'(?:died: (.*?)\n)?'
    r'--- DATA ---\n(.*?)\n\n',
    re.S)


def write_step(log_file, *, step, frame_name, action, plan, reasoning, how,
               outcome, reward, done, achievements, died=None, user_text=None):
    """One step's block in the shared log.txt format: just what a human needs
    to follow along (plan/reasoning/outcome/achievement count, plus a death
    cause when this step killed the agent) with a compact --- DATA --- line
    for build_video to parse back out.

    frame_name is None when no frame was saved this step (e.g. eval_crafter.py's
    --dump-frames only saves every Nth step); build_video skips those for video.
    """
    log_file.write(f'{"=" * 80}\nSTEP {step}  (frame: {frame_name or "none"})\n{"=" * 80}\n\n')
    if user_text is not None:
        log_file.write(f'--- USER PROMPT ---\n{user_text}\n\n')
    log_file.write(f'plan: {plan}\nreasoning: {reasoning}\noutcome: {outcome}\n'
                    f'achievements: {achievements}\n')
    if died:
        log_file.write(f'died: {died}\n')
    data = json.dumps({'action': action, 'how': how, 'reward': reward, 'done': done})
    log_file.write(f'--- DATA ---\n{data}\n\n')
    log_file.flush()


def parse_log(log_path):
    """Inverse of write_step: log.txt -> list of step dicts."""
    text = Path(log_path).read_text()
    steps = []
    for m in STEP_RE.finditer(text):
        step, frame, plan, reasoning, outcome, achievements, died, data_json = m.groups()
        data = json.loads(data_json)
        steps.append({
            'step': int(step), 'frame': None if frame == 'none' else frame,
            'plan': plan.strip(), 'reasoning': ' '.join(reasoning.split()),
            'outcome': outcome.strip(), 'achievements': int(achievements),
            'died': died.strip() if died else None,
            'action': data['action'], 'how': data['how'],
            'reward': data['reward'], 'done': data['done'],
        })
    return steps


def new_run_dir(base, prefix='run'):
    """base/prefix_<timestamp>/, so repeated runs never collide or overwrite."""
    run_dir = Path(base) / f'{prefix}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
