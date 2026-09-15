"""Turning a run's log.txt plus its saved frames into an annotated video:
each frame with its step/action/plan/reasoning burned in below it."""
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.run_log import parse_log

PANEL_HEIGHT = 260
PANEL_BG = (15, 15, 15)


def load_font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def make_composite(frame_path, entry, font, small_font):
    """Annotated frame the model saw, with its step/action/plan/reasoning below it."""
    image = Image.open(frame_path).convert('RGB')
    canvas = Image.new('RGB', (image.width, image.height + PANEL_HEIGHT), PANEL_BG)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    wrap_width = max(40, image.width // 7)

    y = image.height + 8
    draw.text((8, y), f"step {entry['step']}  action: {entry['action']}  "
                       f"({entry['how']})  reward: {entry['reward']:+.2f}",
              fill=(255, 220, 0), font=font)
    y += 24
    for line in textwrap.wrap(f"outcome: {entry['outcome']}", wrap_width)[:2]:
        draw.text((8, y), line, fill=(180, 210, 255), font=small_font)
        y += 18
    for line in textwrap.wrap(f"plan: {entry['plan']}", wrap_width)[:2]:
        draw.text((8, y), line, fill=(180, 255, 180), font=small_font)
        y += 18
    for line in textwrap.wrap(f"reasoning: {entry['reasoning']}", wrap_width)[:6]:
        draw.text((8, y), line, fill=(230, 230, 230), font=small_font)
        y += 18
    return canvas


def build_video(run_dir, fps=3, log_name='log.txt'):
    """log.txt + whatever frame_*.png exist in run_dir -> composite/ + video.mp4.

    Steps with no saved frame are skipped (nothing to show); this is expected
    when frames were only dumped every N steps.
    """
    run_dir = Path(run_dir)
    steps = parse_log(run_dir / log_name)
    steps = [s for s in steps if s['frame'] and (run_dir / s['frame']).exists()]
    if not steps:
        print(f'No frames to build a video from in {run_dir}.')
        return None

    composite_dir = run_dir / 'composite'
    composite_dir.mkdir(exist_ok=True)
    font, small_font = load_font(18), load_font(14)

    video_path = run_dir / 'video.mp4'
    writer = imageio.get_writer(video_path, fps=fps)
    for entry in steps:
        composite = make_composite(run_dir / entry['frame'], entry, font, small_font)
        composite.save(composite_dir / Path(entry['frame']).name)
        writer.append_data(np.array(composite))
    writer.close()

    print(f'{len(steps)} frames -> {composite_dir}/ and {video_path}')
    return video_path


def save_raw_video(frames, path, fps):
    """Plain (unannotated) gameplay clip from a list of raw observation frames."""
    imageio.mimsave(path, frames, fps=fps)
    return path
