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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/quant_geometry"))
    parser.add_argument("--fragment", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stories = build_geometry_stories()
    for story in stories:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(story.title, fontsize=19)
        if story.key == "deformation":
            fig.supxlabel("Two routes: x → xW (FP target), or x → xWq → (xWq)R (corrected)", fontsize=13)
        all_points = np.concatenate([s.numpy() for s in story.states])
        radius = np.abs(all_points).max() * 1.18
        for panel, (ax, index) in enumerate(zip(axes.flat, story.storyboard_indices), start=1):
            state = story.states[index].numpy()
            reference = story.reference_indices[index]
            targets = story.states[reference].numpy() if reference is not None else None
            for geometry, color, alpha, style in [(targets, "#64748b", .55, "--"), (state, "#2563eb", .55, "-")]:
                if geometry is None:
                    continue
                offset = 0
                for j, length in enumerate(story.path_lengths):
                    line = geometry[offset:offset+length]
                    ax.plot(line[:, 0], line[:, 1], style, color=color,
                            linewidth=2.8 if j == len(story.path_lengths)-1 else .8, alpha=alpha)
                    offset += length
            coincident = np.linalg.norm(state[-2]-state[-1]) < 1e-10
            for j, (point, color, marker) in enumerate(zip(state[-2:], ["#d97706", "#9333ea"], ["o", "s"])):
                ax.scatter(*point, c=color, marker=marker, s=80, zorder=6)
                if not coincident:
                    ax.annotate("AB"[j], point, xytext=(9, 10 if j == 0 else -17), textcoords="offset points", color=color, fontsize=13)
                if targets is not None and np.linalg.norm(point-targets[-2+j]) > 1e-6:
                    ax.annotate("", xy=targets[-2+j], xytext=point,
                                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})
            if coincident:
                ax.annotate("A = B", state[-2], xytext=(9, -17), textcoords="offset points", color="#0f172a", fontsize=13)
            ax.axhline(0, color="#cbd5e1", linewidth=.8)
            ax.axvline(0, color="#cbd5e1", linewidth=.8)
            title = story.labels[index]
            if story.key == "deformation":
                title = f"{panel}. {title}\n" + (
                    "The shape before either map",
                    "Target computed directly from the input",
                    "Use Wq instead of W on the same input",
                    "Same coordinates as panel 2; a different route",
                )[panel - 1]
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal",
                   title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
        fig.savefig(args.output_dir / f"geometry_{story.key}.png", dpi=140, facecolor="white")
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
