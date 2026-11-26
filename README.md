# Walk in the Field

A procedurally generated 15-second Blender animation: a walker in a bucket hat, flannel shirt,
cargo pants and hiking boots strolls along a dirt road through a golden field, looks up as a
bird flies overhead, then takes a sip of coffee.

Everything — terrain, crops, trees, character, rig, animation, camera and render settings — is
built from code in [make_walk_in_field.py](make_walk_in_field.py). No external assets are needed.

## Requirements

- [Blender](https://www.blender.org/download/) 5.3 (uses only Blender's bundled Python modules:
  `bpy`, `bmesh`, `mathutils`)

## Usage

Run from PowerShell in this folder:

```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"

# Build the scene and save it to walk_in_field.blend
& $blender -b -P make_walk_in_field.py

# Quick low-resolution test stills of selected frames -> render/still_FFF.png
& $blender -b -P make_walk_in_field.py -- --still 52,208,325 --preview

# Full 1920x1080 animation -> render/walk_in_field.mp4
& $blender -b -P make_walk_in_field.py -- --render
```

| Option | Description |
| --- | --- |
| `--still F[,F..]` | Render the given frame(s) to `render/still_FFF.png` |
| `--render` | Render the full animation to `render/walk_in_field.mp4` |
| `--preview` | Half resolution and fewer samples for quick tests |

Without `--still` or `--render`, the scene is only built and saved to `walk_in_field.blend`.

## Timeline (30 fps, 450 frames)

| Time | Action |
| --- | --- |
| 0–15 s | Walk along the road, tracked by a slowly pushing-in camera |
| 5–9 s | A bird flies overhead; the walker follows it with his head |
| ~9.5–11.8 s | He raises the cup, tips his head back and takes a sip |
