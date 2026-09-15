
# Setup

```
    bash ./setup.sh
    conda init
    conda activate temporal
    pip install -r requirements.txt
```

Model choices
smoke: 8B
mid: 12B 
full: 26B
dense: 31B

Smoke test:
```
python eval_crafter.py --preset full --manual text_files/crafter_info.txt --smoke
```
Final run:
```
python eval_crafter.py --preset full --manual text_files/crafter_info.txt \
  --total-steps 20000 --checkpoint-every 5000
```


# Initial experiments

A dummy run was done to understand if the 12B model was able to understand the environment.

The system had access to a file which had all the important information from the original
Crafter paper (`crafter_info.txt`). At each step the model gets to see past history of prompts
and actions (in this run it was 16): the model will see the last 4 reasoning steps and the last
16 action-output pairs. Only the current image is sent to the model.

## Modifications to the input space

**a. Grid overlay on the frame**
Each frame is divided into a 9x7 canvas, and each grid cell is named A1--I7. With this, the
model was able to identify the grid location clearly. This might also be a bad bias — the
position of the grid for other cells keeps changing, so the model has to interpolate from past
reasoning how they changed, and all that might create confusion.

**b. History**
Last `--history` steps (default 16), where only the most recent `--detail-steps` (default 4)
keep full reasoning text; older ones collapse to `action -> outcome`. This is deliberate — only
the current frame is ever sent as an image, past frames are never re-fed (the module docstring
cites EmbodiedBench/FindingDory results showing that hurts performance).

**c. Observation text**
Grid table + facing direction + what tile you're about to act on + vitals
(health/food/drink/energy) + inventory + a landmark-memory line (materials seen earlier, given as
relative left/right/up/down offsets, since Crafter's window is small and things scroll
off-screen).

**d. System prompt**

> Your goal is to unlock as many achievements as possible before you die.
>
> 1. An image of your surroundings, overlaid with a white grid. Each tile is labelled
>    in yellow, from A1 in the top-left to I7 in the bottom-right.
> 2. A text grid listing the contents of those same labelled tiles. The image and the
>    text describe the SAME nine-by-seven tiles using the SAME labels. When they seem to
>    disagree, trust the text.
> 3. Your vitals, inventory, facing direction, and the tile you are about to act on.
> 4. A short memory of useful things you walked past earlier, and they may now be off-screen.
> 5. Your recent steps, each with what actually happened as a result.
>
> You are always at E4, the centre tile. The world scrolls around you; you never
> leave E4. Moving left means the whole map shifts right.
>
> Answer in exactly this format and nothing else:
>
> ```
> <plan>your current short-term goal, a few words</plan>
> <reasoning>
> What you see, using tile labels. Whether your recent actions worked. Why this action
> serves the plan. Two or three sentences.
> </reasoning>
> <action>one_action_name_from_the_list</action>
> ```
