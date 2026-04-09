"""
Cycle 326 — Item 339: Encoder/Decoder boundary vs b2→b3 eff_rank drop.

Reads config JSON and NPZ checkpoint directly (no subprocess needed — only inspecting
config and weight structure, not running model forward passes).

Key question: does the b2→b3 eff_rank collapse (86→4.89) coincide with the
encoder/decoder boundary?
"""
import json, numpy as np
from pathlib import Path

CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
SUMMARY_JSON = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_summary.json"
CYCLE_318_EFF_RANK = {
    0: 187.61, 1: 160.08, 2: 86.05, 3: 4.89, 4: 38.30, 5: 39.96,
    6: 28.21, 7: 23.23, 8: 30.78, 9: 28.85, 10: 21.55, 11: 13.40, 12: 6.94,
}
OUT_DIR = Path("autoresearch/results")

def main():
    # Load config
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    config = summary["config"]

    num_layers = config["num_layers"]  # 13
    num_encoder_layers_raw = config["num_encoder_layers"]  # 0 => auto
    num_encoder_layers = num_encoder_layers_raw if num_encoder_layers_raw > 0 else num_layers // 2
    num_decoder_layers = num_layers - num_encoder_layers
    boundary_block = num_encoder_layers  # first decoder block index

    print(f"num_layers: {num_layers}")
    print(f"num_encoder_layers (config): {num_encoder_layers_raw}  =>  effective: {num_encoder_layers}")
    print(f"num_decoder_layers: {num_decoder_layers}")
    print(f"Encoder blocks: 0..{num_encoder_layers - 1}")
    print(f"Decoder blocks: {num_encoder_layers}..{num_layers - 1}")
    print(f"Boundary (first decoder block): {boundary_block}")

    # Load skip_weights from NPZ
    data = np.load(CHECKPOINT, allow_pickle=True)
    skip_weights = data["skip_weights"].astype(np.float32)  # (6, 640)
    print(f"\nskip_weights shape: {skip_weights.shape}")
    skip_norms = {i: float(np.linalg.norm(skip_weights[i])) for i in range(skip_weights.shape[0])}
    print("Per-skip Frobenius norms (skip[i] connects encoder block i → decoder block):")
    for i, norm in skip_norms.items():
        print(f"  skip[{i}]: {norm:.4f}")

    # Cross-reference with cycle 318 eff_rank
    drop_block = 3  # b2→b3: block 3 is the destination of the drop
    distance_from_boundary = boundary_block - drop_block
    print(f"\nb2→b3 eff_rank drop: block {drop_block} (eff_rank {CYCLE_318_EFF_RANK[2]:.2f} → {CYCLE_318_EFF_RANK[3]:.2f})")
    print(f"Boundary block: {boundary_block}")
    print(f"Distance from drop block to boundary: {distance_from_boundary} blocks")

    verdict = "REFUTED-no-correlation" if distance_from_boundary >= 2 else "CONFIRMED"
    print(f"\nVerdict: {verdict}")
    print(f"Reason: block 3 is deep inside the encoder (3 blocks from boundary at block 6).")

    # Write JSON result
    result = {
        "checkpoint": CHECKPOINT,
        "num_layers": num_layers,
        "num_encoder_layers_raw": num_encoder_layers_raw,
        "num_encoder_layers_effective": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "boundary_block": boundary_block,
        "skip_weights_shape": list(skip_weights.shape),
        "skip_norms": skip_norms,
        "cycle_318_eff_rank_profile": CYCLE_318_EFF_RANK,
        "b2_to_b3_drop_block": drop_block,
        "distance_drop_to_boundary": distance_from_boundary,
        "verdict": verdict,
    }
    out_json = OUT_DIR / "cycle_326_enc_dec_split.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nWrote: {out_json}")
    return result


if __name__ == "__main__":
    main()
