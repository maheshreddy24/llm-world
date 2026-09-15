"""Static config: model presets, the action set, and achievement metadata."""
import re

import crafter

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
