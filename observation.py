"""Input/output encoding for the Crafter VLM agent.

Design, in one paragraph: the image and the text describe the same 9x7 tile grid
using the same labels (A1..I7), burned into the frame and printed in the text.
That gives the model one referring vocabulary across both channels, so mapping
"the tree I see" to "which key to press" is a lookup rather than a spatial
inference. This is Set-of-Mark prompting (Yang et al.) applied to a tile game.

Everything is screen-relative (left/right/up/down) to match the action names.
No compass directions anywhere -- a north/west vocabulary forces a translation
step on every decision and that is where small models drop tiles.

Crafter's world coordinates are (x, y) with y increasing DOWNWARD, so 'up' is
y-1. Getting this backwards is silent and costly.

Drop-in: import format_observation, annotate_frame, MASTER_PROMPT, STEP_PROMPT
into eval_crafter.py and replace read_state/format_state.
"""
from PIL import Image, ImageDraw

# 9 wide x 7 tall world view (crafter's default view is 9x9 with 2 HUD rows).
VIEW_W, VIEW_H = 9, 7
COLS = 'ABCDEFGHI'
ROWS = '1234567'
PLAYER_COL, PLAYER_ROW = 4, 3          # zero-based centre of the 9x7 window
PLAYER_LABEL = COLS[PLAYER_COL] + ROWS[PLAYER_ROW]   # 'E4'

VITALS = ('health', 'food', 'drink', 'energy')
MAX_VITAL = 9

# Screen-relative only. Crafter pos is (x, y), y grows downward.
FACING_NAMES = {(-1, 0): 'left', (1, 0): 'right', (0, -1): 'up', (0, 1): 'down'}
FACING_TARGET = {(-1, 0): (-1, 0), (1, 0): (1, 0), (0, -1): (0, -1), (0, 1): (0, 1)}

# Materials worth remembering across the episode once seen.
LANDMARK_TILES = {'water', 'stone', 'coal', 'iron', 'diamond', 'tree', 'table', 'furnace', 'lava'}


def label(col, row):
    """Grid label for zero-based window coordinates."""
    return f'{COLS[col]}{ROWS[row]}'


def cell_contents(env, world_pos):
    """Name of what occupies a world cell: object if present, else material."""
    try:
        material, obj = env._world[tuple(world_pos)]
    except Exception:
        return 'unknown'
    if obj is not None:
        return type(obj).__name__.lower()
    return material or 'unknown'


def scan_window(env):
    """Return {label: contents} for the visible 9x7 window, plus landmark sightings."""
    px, py = (int(v) for v in env._player.pos)
    window, landmarks = {}, {}
    for row in range(VIEW_H):
        for col in range(VIEW_W):
            dx, dy = col - PLAYER_COL, row - PLAYER_ROW
            name = cell_contents(env, (px + dx, py + dy))
            tag = label(col, row)
            window[tag] = name
            if name in LANDMARK_TILES:
                landmarks.setdefault(name, (px + dx, py + dy))
    return window, landmarks


def render_grid(window):
    """The text twin of the annotated image. ~180 tokens, worth every one."""
    width = max(len(v) for v in window.values()) + 1
    width = max(width, 8)
    header = '    ' + ''.join(c.center(width) for c in COLS)
    lines = [header]
    for row in range(VIEW_H):
        cells = []
        for col in range(VIEW_W):
            name = window[label(col, row)]
            if col == PLAYER_COL and row == PLAYER_ROW:
                name = '[YOU]'
            cells.append(name.center(width))
        lines.append(f' {ROWS[row]}  ' + ''.join(cells))
    return '\n'.join(lines)


def annotate_frame(frame, image_size, draw_labels=True):
    """Burn the same A1..I7 labels onto the rendered frame.

    Crafter draws the 9x9 view into a square canvas with the bottom 2 rows as
    HUD, so the world occupies the top 7/9 of the image.
    """
    image = Image.fromarray(frame).convert('RGB')
    if not draw_labels:
        return image
    tile = image_size / 9.0          # crafter's view is 9x9 including HUD rows
    pen = ImageDraw.Draw(image)
    for col in range(VIEW_W + 1):
        x = col * tile
        pen.line([(x, 0), (x, VIEW_H * tile)], fill=(255, 255, 255), width=1)
    for row in range(VIEW_H + 1):
        y = row * tile
        pen.line([(0, y), (VIEW_W * tile, y)], fill=(255, 255, 255), width=1)
    for row in range(VIEW_H):
        for col in range(VIEW_W):
            pen.text((col * tile + 2, row * tile + 1), label(col, row),
                     fill=(255, 255, 0))
    return image


def read_state(env, seen=None):
    """Vitals, inventory, facing, faced cell, visible grid, landmark memory."""
    player = env._player
    inventory = dict(player.inventory)
    vitals = {k: inventory.pop(k, 0) for k in VITALS}
    items = {k: v for k, v in inventory.items() if v > 0}
    facing = tuple(int(v) for v in player.facing)
    pos = tuple(int(v) for v in player.pos)

    window, landmarks = scan_window(env)
    if seen is not None:
        for name, where in landmarks.items():
            seen.setdefault(name, where)      # keep the FIRST sighting

    dx, dy = FACING_TARGET.get(facing, (0, 0))
    faced_label = label(PLAYER_COL + dx, PLAYER_ROW + dy) \
        if 0 <= PLAYER_COL + dx < VIEW_W and 0 <= PLAYER_ROW + dy < VIEW_H else None
    faced = window.get(faced_label, cell_contents(env, (pos[0] + dx, pos[1] + dy)))

    return {
        'vitals': vitals, 'inventory': items, 'facing': facing, 'pos': pos,
        'window': window, 'faced': faced, 'faced_label': faced_label,
        'achievements': {k for k, v in player.achievements.items() if v > 0},
    }


def format_landmarks(seen, pos):
    """Compact memory of things walked past. Crafter's window is tiny; this matters."""
    if not seen:
        return 'You have not yet seen anything worth remembering.'
    px, py = pos
    parts = []
    for name, (lx, ly) in sorted(seen.items()):
        dx, dy = lx - px, ly - py
        horizontal = f'{abs(dx)} left' if dx < 0 else (f'{abs(dx)} right' if dx else '')
        vertical = f'{abs(dy)} up' if dy < 0 else (f'{abs(dy)} down' if dy else '')
        where = ', '.join(p for p in (horizontal, vertical) if p) or 'right here'
        parts.append(f'{name} ({where})')
    return 'Remembered from earlier: ' + '; '.join(parts) + '.'


def format_observation(state, seen):
    vitals = ', '.join(f'{k} {state["vitals"].get(k, 0)}/{MAX_VITAL}' for k in VITALS)
    inventory = ', '.join(f'{k} {v}' for k, v in sorted(state['inventory'].items())) or 'empty'
    facing = FACING_NAMES.get(state['facing'], '?')
    faced_at = f' ({state["faced_label"]})' if state['faced_label'] else ''
    return (
        f'{render_grid(state["window"])}\n\n'
        f'You are [YOU] at {PLAYER_LABEL}. You are facing {facing}, so "do" and every '
        f'"place_" action will act on{faced_at}, which currently contains: {state["faced"]}.\n'
        f'Status: {vitals}\n'
        f'Inventory: {inventory}\n'
        f'{format_landmarks(seen, state["pos"])}')


MASTER_PROMPT = """You are an expert agent playing Crafter, a 2D survival and crafting game.
Your goal is to unlock as many achievements as possible before you die.

## What you receive each step

1. An image of your surroundings, overlaid with a white grid. Each tile is labelled
   in yellow, from A1 in the top-left to I7 in the bottom-right.
2. A text grid listing the contents of those same labelled tiles. The image and the
   text describe the SAME nine-by-seven tiles using the SAME labels. When they seem to
   disagree, trust the text.
3. Your vitals, inventory, facing direction, and the tile you are about to act on.
4. A short memory of useful things you walked past earlier, and they may now be off-screen.
5. Your recent steps, each with what actually happened as a result.

You are always at E4, the centre tile. The world scrolls around you; you never
leave E4. Moving left means the whole map shifts right.

## Directions

Everything is described from your point of view on screen: left, right, up, down.
Column A is furthest left, column I furthest right. Row 1 is the top, row 7 the bottom.
To reach a tile in column D when you are in column E, move_left. To reach row 2 from
row 4, move_up twice. There are no compass directions in this game.

## The rule book

{manual}

## Actions

Choose exactly one of these 17 names each step:

{actions}

An action whose requirements are not met does nothing and wastes the step.

## Facing

"do" and every "place_" action affect ONLY the single tile directly in front of you,
which is named for you each step. Facing is set by movement: pressing a move key
towards an obstacle turns you to face it without moving you. So to chop a tree at D4
while standing at E4, press move_left once (you turn to face it, blocked by the tree)
and then "do".

## What to prioritise

Survival first. If drink or food drops below about 3, go fix it before anything else:
drink by facing water and using "do", eat by killing a cow with "do". Health only
regenerates while the other three bars are healthy.

Otherwise work down the tech tree in order. Collect wood from trees. Place a table.
Craft a wood pickaxe and wood sword. Mine stone. Craft stone tools. Place a furnace.
Find coal and iron. Craft iron tools. Look for diamond. Each of these is an achievement
and shallow ones are worth far more than a failed attempt at a deep one.

At night zombies spawn. Wall yourself in with place_stone on all open sides, then sleep.

## Reading your own history

Each past step tells you what actually changed. Take this seriously. If a result says
"nothing changed", the action had no effect and repeating it will do nothing again.
If two or three consecutive results say nothing changed, your current approach is
wrong: you are facing the wrong tile, you are missing a required tool or material, or
you are not standing near a table. Say so in your reasoning and try something different.
Do not repeat a failing action.

## How you must answer

Answer in exactly this format and nothing else:

<plan>your current short-term goal, a few words</plan>
<reasoning>
What you see, using tile labels. Whether your recent actions worked. Why this action
serves the plan. Two or three sentences.
</reasoning>
<action>one_action_name_from_the_list</action>

Carry the plan forward across steps unless it is achieved or clearly impossible, then
replace it. The action name must be written exactly as it appears in the list."""


STEP_PROMPT = """Step {step}.

{observation}

{history}

Your plan from last step was: {plan}

Choose your next action."""


NO_HISTORY = 'This is your first step, so you have no history yet.'
NO_PLAN = 'nothing yet, decide one now'
HISTORY_HEADER = 'Your recent steps, oldest first:'


def format_history(history, detail_steps=4):
    """Recent steps in full; older ones compressed to action and result.

    Full reasoning for every step burns tokens and buries the signal. The
    outcome line is what stops the agent looping, so that is what is kept.
    """
    if not history:
        return NO_HISTORY
    lines = [HISTORY_HEADER]
    cutoff = len(history) - detail_steps
    for index, entry in enumerate(history):
        head = f"step {entry['step']} | {entry['action']} -> {entry['outcome']}"
        if index >= cutoff:
            lines.append(f"{head} | plan: {entry.get('plan', '-')} | {entry['reasoning']}")
        else:
            lines.append(head)
    return '\n'.join(lines)
