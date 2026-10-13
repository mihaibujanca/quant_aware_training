"""Generate geometry-first storyboards and exact state data for animation.

.venv/bin/python -m experiments.quant_geometry_visual_story
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aleph.qgeom.geometry_story import build_geometry_stories
from experiments import quant_geometry_style as style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/quant_geometry"))
    parser.add_argument("--fragment", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stories = build_geometry_stories()
    # Panel kinds follow experiments/quant_geometry_style.py: what state each shape is in.
    panels = {
        "deformation": [
            (0, "input", False, "1. Input $x$"),
            (1, "fp", False, "2. FP target: $xW$"),
            (2, "q", True, "3. Quantized: $xW_q$"),
            (3, "repaired", True, "4. Corrected: $(xW_q)R$, equal to the target"),
        ],
        "relu": [
            (0, "q", False, "1. Quantized preactivation $z_q=xW_q$"),
            (2, "fp", False, "2. FP target: $y=\\phi(z_qR)$"),
            (3, "q", True, "3. ReLU first: $h=\\phi(z_q)$\nA and B merge at zero"),
            (4, "repaired", True, "4. Then correct: $hR$\nA and B stay merged"),
        ],
        "loss": [
            (1, "fp", False, "1. FP target: $xW$"),
            (2, "q", True, "2. Quantized: $xW_q$\nthe square collapses to a line"),
            (3, "repaired", True, "3. Corrected: $(xW_q)R$\nstill a line"),
            (3, "repaired", True, "4. Adding $xU$ (purple) reaches the target"),
        ],
    }
    for story in stories:
        fp_index = {"deformation": 1, "relu": 2, "loss": 1}[story.key]
        fp_state = story.states[fp_index].numpy()
        fig, axes = plt.subplots(2, 2, figsize=(11, 10.6), constrained_layout=True)
        fig.suptitle(story.title, fontsize=19)
        all_points = np.concatenate([s.numpy() for s in story.states])
        radius = np.abs(all_points).max() * 1.18
        bounds = ((-radius, radius), (-radius, radius))
        names = "AB"
        for ax, (index, kind, overlay, title) in zip(axes.flat, panels[story.key]):
            state = story.states[index].numpy()
            if overlay:
                style.shape(ax, fp_state, story.path_lengths, "fp")
            style.shape(ax, state, story.path_lengths, kind)
            style.named(ax, state[-2:], names, kind)
            if story.key == "loss" and title.startswith("4."):
                for j in range(2):
                    style.arrow(ax, state[-2+j], fp_state[-2+j], "U")
            style.axes_style(ax, bounds, title,
                             xlabel="Input coordinate 1" if kind == "input" else "Coordinate 1",
                             ylabel="Input coordinate 2" if kind == "input" else "Coordinate 2")
        kinds = ["input", "fp", "q", "repaired"] if story.key == "deformation" else ["fp", "q", "repaired"]
        handles = [style.state_handle(k) for k in kinds]
        if story.key == "loss":
            handles.append(style.term_handle("U", "$xU$"))
        style.legend(fig, handles)
        fig.savefig(args.output_dir / f"geometry_{story.key}.png", dpi=140, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    payload = {"stories": [s.payload() for s in stories]}
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (args.output_dir / "geometry_story_states.json").write_text(serialized + "\n")
    if args.fragment:
        template = Path("animations/quant_geometry_story.html").read_text()
        args.fragment.parent.mkdir(parents=True, exist_ok=True)
        args.fragment.write_text(template.replace("__GEOMETRY_DATA__", serialized))
    print(f"Generated three geometric storyboards and {len(serialized)} bytes of exact-state animation data")


if __name__ == "__main__":
    main()
