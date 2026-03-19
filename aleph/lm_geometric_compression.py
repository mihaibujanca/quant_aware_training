from __future__ import annotations

import copy
import importlib
import math
import os
import random
import sys

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


class SimpleByteTokenizer:
    """Minimal tokenizer for offline tests and random GPT-style models."""

    def __init__(self, vocab_size: int = 256, bos_token_id: int | None = None):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2.")
        self.vocab_size = int(vocab_size)
        self.bos_token_id = bos_token_id

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8", errors="ignore")
        return [int(byte) % self.vocab_size for byte in raw]

    def decode(self, tokens: Iterable[int]) -> str:
        vals = [int(tok) % 256 for tok in tokens]
        return bytes(vals).decode("utf-8", errors="ignore")

    @property
    def bos_token(self) -> str | None:
        return None if self.bos_token_id is None else "<|bos|>"

    @property
    def eos_token_id(self) -> int | None:
        return None


@dataclass(frozen=True)
class TeacherLoadSpec:
    source: str = "hf_pretrained"
    hf_model_name: str = "gpt2"
    hf_config_overrides: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str | None = None
    nanochat_repo: str | None = None
    nanochat_source: str = "base"
    nanochat_model_tag: str | None = None
    nanochat_step: int | None = None
    device: str = "cpu"
    dataset_name: str = "climbmix-400b-shuffle"


@dataclass(frozen=True)
class ProbeCorpusSpec:
    sequence_length: int = 128
    probe_train_sequences: int = 64
    probe_holdout_sequences: int = 32
    eval_val_sequences: int = 32
    text_chunks: tuple[str, ...] = ()
    text_path: str | None = None
    parquet_dir: str | None = None
    shuffle_seed: int = 42
    max_documents: int = 256
    teacher: TeacherBundle | None = None
    tokenizer: Any | None = None
    dataset_name: str | None = None


@dataclass(frozen=True)
class CompressionBudgetSpec:
    hidden_fraction: float = 0.25
    min_hidden: int = 32
    max_hidden: int | None = None
    per_layer_hidden: dict[int, int] = field(default_factory=dict)
    selection_mode: str = "topk"
    kmeans_seed: int = 42
    dead_threshold: float = 1e-6
    jacobian_samples: int = 8


@dataclass(frozen=True)
class DistillationSpec:
    teacher: TeacherBundle
    corpus: ProbeCorpus
    budget: CompressionBudgetSpec
    layer_ids: tuple[int, ...] | None = None
    batch_size: int = 4
    steps: int = 100
    lr: float = 1e-3
    kl_weight: float = 1.0
    target_ce_weight: float = 0.25
    layer_output_weight: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class LearnedCompressorSpec:
    teacher: TeacherBundle
    corpus: ProbeCorpus
    budget: CompressionBudgetSpec
    layer_ids: tuple[int, ...] | None = None
    code_dim: int = 32
    autoencoder_hidden: int = 128
    epochs: int = 25
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class FfnLayerHandle:
    layer_id: int
    block: nn.Module
    mlp: nn.Module
    c_fc: nn.Module
    c_proj: nn.Module
    activation_kind: str


@dataclass(frozen=True)
class TeacherBundle:
    source: str
    model_family: str
    model_name: str
    dataset_name: str
    model: nn.Module
    tokenizer: Any
    device: torch.device
    max_seq_len: int
    layer_ids: tuple[int, ...]
    activation_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeCorpus:
    dataset_name: str
    sequence_length: int
    probe_train: torch.Tensor
    probe_holdout: torch.Tensor
    eval_val: torch.Tensor
    token_counts: dict[str, int]


@dataclass(frozen=True)
class FfnLayerGeometry:
    layer_id: int
    activation_kind: str
    input_dim: int
    hidden_dim: int
    handle: FfnLayerHandle
    input_sample: torch.Tensor
    output_sample: torch.Tensor
    preactivation_sample: torch.Tensor
    hidden_sample: torch.Tensor
    channel_scores: torch.Tensor
    mean_hidden_activation: torch.Tensor
    std_hidden_activation: torch.Tensor
    mean_hidden_derivative: torch.Tensor
    output_norms: torch.Tensor
    dead_mask: torch.Tensor
    stable_positive_mask: torch.Tensor
    stable_negative_mask: torch.Tensor
    jacobian_spectrum: torch.Tensor
    stable_stats: dict[str, float]


@dataclass(frozen=True)
class FfnGeometryAtlas:
    model_family: str
    dataset_name: str
    sequence_length: int
    layer_geometries: tuple[FfnLayerGeometry, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressedFfnLayer:
    layer_id: int
    activation_kind: str
    source_hidden_dim: int
    compressed_hidden_dim: int
    c_fc_weight: torch.Tensor
    c_fc_bias: torch.Tensor | None
    c_proj_weight: torch.Tensor
    c_proj_bias: torch.Tensor | None
    selected_indices: torch.Tensor | None
    source_param_count: int
    compressed_param_count: int
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressedFfnPack:
    method: str
    model_family: str
    dataset_name: str
    layers: tuple[CompressedFfnLayer, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionRunResult:
    method: str
    metrics: dict[str, float]
    layer_metrics: tuple[dict[str, float], ...]
    pack: CompressedFfnPack
    metadata: dict[str, Any] = field(default_factory=dict)


class CompressedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation_kind: str,
        *,
        c_fc_bias: bool,
        c_proj_bias: bool,
    ):
        super().__init__()
        self.activation_kind = activation_kind
        self.c_fc = nn.Linear(input_dim, hidden_dim, bias=c_fc_bias)
        self.c_proj = nn.Linear(hidden_dim, output_dim, bias=c_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = _apply_activation(x, self.activation_kind)
        x = self.c_proj(x)
        return x


class DescriptorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, code_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon


def _device_for(device_spec: str) -> torch.device:
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_spec)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        try:
            return list(tokenizer.encode(text))
        except TypeError:
            pass
    if callable(tokenizer):
        encoded = tokenizer(text)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return list(encoded["input_ids"])
        if isinstance(encoded, (list, tuple)):
            return list(encoded)
    raise TypeError("Tokenizer does not expose a supported encode interface.")


def _bos_token_id(tokenizer: Any) -> int | None:
    for attr in ("bos_token_id",):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    if hasattr(tokenizer, "tokenizer"):
        value = getattr(tokenizer.tokenizer, "bos_token_id", None)
        if value is not None:
            return int(value)
    return None


def _read_text_chunks(spec: ProbeCorpusSpec) -> tuple[list[str], list[str]]:
    docs: list[str] = []
    if spec.text_chunks:
        docs.extend(str(chunk) for chunk in spec.text_chunks if str(chunk).strip())
    if spec.text_path is not None:
        text = Path(spec.text_path).read_text(encoding="utf-8")
        docs.extend(chunk.strip() for chunk in text.split("\n\n") if chunk.strip())
    if spec.parquet_dir is not None:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading nanochat parquet shards requires pyarrow. Install project dependencies again."
            ) from exc
        parquet_paths = sorted(Path(spec.parquet_dir).glob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet shards found in {spec.parquet_dir}.")
        train_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
        val_paths = parquet_paths[-1:] if len(parquet_paths) > 1 else parquet_paths
        train_docs: list[str] = []
        val_docs: list[str] = []
        for bucket, paths in ((train_docs, train_paths), (val_docs, val_paths)):
            for path in paths:
                pf = pq.ParquetFile(path)
                for row_group in range(pf.num_row_groups):
                    texts = pf.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
                    bucket.extend(str(text) for text in texts if str(text).strip())
                    if len(bucket) >= spec.max_documents:
                        break
                if len(bucket) >= spec.max_documents:
                    break
        return train_docs[: spec.max_documents], val_docs[: spec.max_documents]
    if not docs:
        raise ValueError("No corpus source provided. Set text_chunks, text_path, or parquet_dir.")
    rng = random.Random(spec.shuffle_seed)
    rng.shuffle(docs)
    split = max(1, int(0.8 * len(docs)))
    train_docs = docs[:split]
    val_docs = docs[split:] or docs[-1:]
    return train_docs, val_docs


def _token_windows(
    docs: list[str],
    tokenizer: Any,
    sequence_length: int,
    num_sequences: int,
) -> torch.Tensor:
    bos = _bos_token_id(tokenizer)
    token_stream: list[int] = []
    for doc in docs:
        if bos is not None:
            token_stream.append(bos)
        token_stream.extend(_encode_text(tokenizer, doc))
        if len(token_stream) >= (sequence_length + 1) * num_sequences:
            break
    if len(token_stream) < sequence_length + 1:
        raise ValueError("Corpus does not contain enough tokens for one sequence window.")
    usable = min(len(token_stream), (sequence_length + 1) * num_sequences)
    usable -= usable % (sequence_length + 1)
    windowed = torch.tensor(token_stream[:usable], dtype=torch.long)
    return windowed.view(-1, sequence_length + 1)


def _load_nanochat_from_repo(spec: TeacherLoadSpec, device: torch.device) -> tuple[nn.Module, Any, dict[str, Any]]:
    if spec.nanochat_repo is None:
        raise ValueError("nanochat_repo is required for source='nanochat_local'.")
    repo_path = os.fspath(Path(spec.nanochat_repo).expanduser().resolve())
    if not Path(repo_path).exists():
        raise FileNotFoundError(f"nanochat repo not found: {repo_path}")
    sys.path.insert(0, repo_path)
    try:
        checkpoint_manager = importlib.import_module("nanochat.checkpoint_manager")
        model, tokenizer, meta = checkpoint_manager.load_model(
            spec.nanochat_source,
            device,
            phase="eval",
            model_tag=spec.nanochat_model_tag,
            step=spec.nanochat_step,
        )
    finally:
        sys.path = [path for path in sys.path if path != repo_path]
    return model, tokenizer, meta


def load_nanochat_teacher(spec: TeacherLoadSpec) -> TeacherBundle:
    device = _device_for(spec.device)
    if spec.source == "hf_pretrained":
        tokenizer_name = spec.tokenizer_name or spec.hf_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.hf_model_name).to(device)
        model.eval()
        max_seq_len = int(
            getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_position_embeddings", None)
            or 1024
        )
        model_family = "gpt2"
        model_name = spec.hf_model_name
        activation_kind = "gelu"
        metadata = {"source": "huggingface_pretrained"}
    elif spec.source == "hf_random":
        config_kwargs = {
            "vocab_size": 256,
            "n_positions": 64,
            "n_ctx": 64,
            "n_embd": 64,
            "n_layer": 2,
            "n_head": 2,
            "n_inner": 256,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        }
        config_kwargs.update(spec.hf_config_overrides)
        config = GPT2Config(**config_kwargs)
        model = GPT2LMHeadModel(config).to(device)
        model.eval()
        tokenizer = SimpleByteTokenizer(vocab_size=config.vocab_size)
        max_seq_len = int(config.n_positions)
        model_family = "gpt2"
        model_name = "gpt2-random"
        activation_kind = "gelu"
        metadata = {"source": "huggingface_random", "config": config.to_dict()}
    elif spec.source == "nanochat_local":
        model, tokenizer, meta = _load_nanochat_from_repo(spec, device)
        max_seq_len = int(meta["model_config"]["sequence_len"])
        model_family = "nanochat"
        model_name = spec.nanochat_model_tag or "nanochat"
        activation_kind = "relu_squared"
        metadata = meta
    else:
        raise ValueError(
            f"Unsupported teacher source {spec.source!r}. "
            "Choose from 'hf_pretrained', 'hf_random', or 'nanochat_local'."
        )
    layer_ids = tuple(handle.layer_id for handle in _extract_ffn_handles(model, activation_kind))
    return TeacherBundle(
        source=spec.source,
        model_family=model_family,
        model_name=model_name,
        dataset_name=spec.dataset_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=max_seq_len,
        layer_ids=layer_ids,
        activation_kind=activation_kind,
        metadata=metadata,
    )


def build_probe_corpus(spec: ProbeCorpusSpec) -> ProbeCorpus:
    tokenizer = spec.tokenizer or (spec.teacher.tokenizer if spec.teacher is not None else None)
    if tokenizer is None:
        raise ValueError("ProbeCorpusSpec requires tokenizer or teacher.")
    train_docs, val_docs = _read_text_chunks(spec)
    probe_train = _token_windows(train_docs, tokenizer, spec.sequence_length, spec.probe_train_sequences)
    remaining_train_docs = train_docs[::-1] if len(train_docs) > 1 else train_docs
    probe_holdout = _token_windows(
        remaining_train_docs,
        tokenizer,
        spec.sequence_length,
        spec.probe_holdout_sequences,
    )
    eval_val = _token_windows(val_docs, tokenizer, spec.sequence_length, spec.eval_val_sequences)
    dataset_name = spec.dataset_name or (spec.teacher.dataset_name if spec.teacher is not None else "custom")
    return ProbeCorpus(
        dataset_name=dataset_name,
        sequence_length=spec.sequence_length,
        probe_train=probe_train,
        probe_holdout=probe_holdout,
        eval_val=eval_val,
        token_counts={
            "probe_train": int(probe_train.numel()),
            "probe_holdout": int(probe_holdout.numel()),
            "eval_val": int(eval_val.numel()),
        },
    )


def _extract_ffn_handles(model: nn.Module, default_activation: str) -> list[FfnLayerHandle]:
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    if blocks is None:
        raise ValueError("Unsupported model architecture: could not locate transformer blocks.")

    handles: list[FfnLayerHandle] = []
    for layer_id, block in enumerate(blocks):
        mlp = getattr(block, "mlp", None) or getattr(block, "ffn", None)
        if mlp is None:
            continue
        c_fc = getattr(mlp, "c_fc", None)
        c_proj = getattr(mlp, "c_proj", None)
        if c_fc is None or c_proj is None:
            continue
        activation_kind = default_activation
        act = getattr(mlp, "act", None)
        if act is not None:
            act_name = act.__class__.__name__.lower()
            if "gelu" in act_name:
                activation_kind = "gelu"
            elif "relu" in act_name:
                activation_kind = "relu"
        handles.append(
            FfnLayerHandle(
                layer_id=layer_id,
                block=block,
                mlp=mlp,
                c_fc=c_fc,
                c_proj=c_proj,
                activation_kind=activation_kind,
            )
        )
    if not handles:
        raise ValueError("No FFN blocks with c_fc/c_proj were found.")
    return handles


def _standard_weight(linear_like: nn.Module) -> torch.Tensor:
    weight = linear_like.weight.detach().cpu()
    if linear_like.__class__.__name__ == "Conv1D":
        return weight.t().contiguous()
    return weight.contiguous()


def _standard_bias(linear_like: nn.Module) -> torch.Tensor | None:
    bias = getattr(linear_like, "bias", None)
    if bias is None:
        return None
    return bias.detach().cpu().contiguous()


def _apply_activation(x: torch.Tensor, activation_kind: str) -> torch.Tensor:
    if activation_kind == "gelu":
        return F.gelu(x)
    if activation_kind == "relu_squared":
        return F.relu(x).square()
    if activation_kind == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported activation_kind {activation_kind!r}")


def _activation_derivative(x: torch.Tensor, activation_kind: str) -> torch.Tensor:
    if activation_kind == "gelu":
        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        return 0.5 * (1.0 + torch.erf(x * inv_sqrt_2)) + x * torch.exp(-0.5 * x.square()) * inv_sqrt_2pi
    if activation_kind == "relu_squared":
        return 2.0 * torch.relu(x)
    if activation_kind == "relu":
        return (x > 0).to(dtype=x.dtype)
    raise ValueError(f"Unsupported activation_kind {activation_kind!r}")


def _param_count(weight: torch.Tensor, bias: torch.Tensor | None) -> int:
    total = int(weight.numel())
    if bias is not None:
        total += int(bias.numel())
    return total


def _mean_jacobian_spectrum(
    weight_in: torch.Tensor,
    weight_out: torch.Tensor,
    preactivations: torch.Tensor,
    activation_kind: str,
    jacobian_samples: int,
) -> torch.Tensor:
    if preactivations.ndim != 2:
        raise ValueError("preactivations must be a [N, H] tensor.")
    if preactivations.shape[0] == 0:
        return torch.zeros(min(weight_out.shape[0], weight_in.shape[1]), dtype=torch.float32)
    sample_count = min(jacobian_samples, preactivations.shape[0])
    spectra = []
    for idx in range(sample_count):
        deriv = _activation_derivative(preactivations[idx], activation_kind)
        jacobian = (weight_out * deriv.unsqueeze(0)) @ weight_in
        spectra.append(torch.linalg.svdvals(jacobian.to(dtype=torch.float32)))
    return torch.stack(spectra, dim=0).mean(dim=0)


@contextmanager
def _collect_ffn_outputs(handles: list[FfnLayerHandle]):
    outputs: dict[int, list[torch.Tensor]] = {handle.layer_id: [] for handle in handles}
    hooks = []
    try:
        for handle in handles:
            hooks.append(
                handle.mlp.register_forward_hook(
                    lambda _module, _inputs, output, *, layer_id=handle.layer_id: outputs[layer_id].append(
                        output.detach().cpu()
                    )
                )
            )
        yield outputs
    finally:
        for hook in hooks:
            hook.remove()


def _iter_lm_batches(sequences: torch.Tensor, batch_size: int, device: torch.device):
    for start in range(0, sequences.shape[0], batch_size):
        batch = sequences[start : start + batch_size].to(device)
        yield batch[:, :-1], batch[:, 1:]


def _sample_tensor(existing: list[torch.Tensor], new_value: torch.Tensor, remaining: int) -> int:
    if remaining <= 0:
        return 0
    flat = new_value.reshape(-1, new_value.shape[-1]).detach().cpu()
    take = min(remaining, flat.shape[0])
    if take > 0:
        existing.append(flat[:take])
    return take


def _forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("Model forward did not return logits or a Hugging Face output object.")


def probe_ffn_geometry(
    teacher: TeacherBundle,
    corpus: ProbeCorpus,
    layer_ids: tuple[int, ...] | None = None,
    *,
    max_tokens_per_layer: int = 2048,
    batch_size: int = 4,
    jacobian_samples: int = 8,
) -> FfnGeometryAtlas:
    all_handles = _extract_ffn_handles(teacher.model, teacher.activation_kind)
    selected_handles = [
        handle for handle in all_handles if layer_ids is None or handle.layer_id in set(layer_ids)
    ]
    layer_geometries: list[FfnLayerGeometry] = []
    teacher.model.eval()
    for handle in selected_handles:
        input_samples: list[torch.Tensor] = []
        output_samples: list[torch.Tensor] = []
        preactivation_samples: list[torch.Tensor] = []
        hidden_samples: list[torch.Tensor] = []

        def hook_fn(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            h = _apply_activation(z, handle.activation_kind).detach()
            remaining = max_tokens_per_layer - sum(item.shape[0] for item in input_samples)
            taken = _sample_tensor(input_samples, x, remaining)
            if taken <= 0:
                return
            _sample_tensor(output_samples, output, taken)
            _sample_tensor(preactivation_samples, z, taken)
            _sample_tensor(hidden_samples, h, taken)

        hook = handle.mlp.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for input_ids, _targets in _iter_lm_batches(corpus.probe_holdout, batch_size, teacher.device):
                    _ = teacher.model(input_ids)
                    if sum(item.shape[0] for item in input_samples) >= max_tokens_per_layer:
                        break
        finally:
            hook.remove()

        x = torch.cat(input_samples, dim=0).to(dtype=torch.float32)
        y = torch.cat(output_samples, dim=0).to(dtype=torch.float32)
        z = torch.cat(preactivation_samples, dim=0).to(dtype=torch.float32)
        h = torch.cat(hidden_samples, dim=0).to(dtype=torch.float32)

        weight_in = _standard_weight(handle.c_fc).to(dtype=torch.float32)
        weight_out = _standard_weight(handle.c_proj).to(dtype=torch.float32)
        deriv = _activation_derivative(z, handle.activation_kind)
        mean_hidden = h.mean(dim=0)
        std_hidden = h.std(dim=0) if h.shape[0] > 1 else torch.zeros_like(mean_hidden)
        mean_deriv = deriv.mean(dim=0)
        output_norms = weight_out.norm(dim=0)
        dead_mask = h.abs().amax(dim=0) <= 1e-6
        stable_positive_mask = (z > 0).all(dim=0)
        stable_negative_mask = (z <= 0).all(dim=0)
        channel_scores = std_hidden * output_norms * (mean_hidden.abs() + mean_deriv.abs() + 1e-6)
        jacobian_spectrum = _mean_jacobian_spectrum(
            weight_in,
            weight_out,
            z,
            handle.activation_kind,
            jacobian_samples,
        )
        stable_stats = {
            "dead_fraction": float(dead_mask.float().mean().item()),
            "stable_positive_fraction": float(stable_positive_mask.float().mean().item()),
            "stable_negative_fraction": float(stable_negative_mask.float().mean().item()),
        }
        layer_geometries.append(
            FfnLayerGeometry(
                layer_id=handle.layer_id,
                activation_kind=handle.activation_kind,
                input_dim=int(x.shape[-1]),
                hidden_dim=int(h.shape[-1]),
                handle=handle,
                input_sample=x,
                output_sample=y,
                preactivation_sample=z,
                hidden_sample=h,
                channel_scores=channel_scores,
                mean_hidden_activation=mean_hidden,
                std_hidden_activation=std_hidden,
                mean_hidden_derivative=mean_deriv,
                output_norms=output_norms,
                dead_mask=dead_mask,
                stable_positive_mask=stable_positive_mask,
                stable_negative_mask=stable_negative_mask,
                jacobian_spectrum=jacobian_spectrum,
                stable_stats=stable_stats,
            )
        )
    return FfnGeometryAtlas(
        model_family=teacher.model_family,
        dataset_name=corpus.dataset_name,
        sequence_length=corpus.sequence_length,
        layer_geometries=tuple(layer_geometries),
        metadata={
            "model_name": teacher.model_name,
            "max_tokens_per_layer": max_tokens_per_layer,
        },
    )


def _layer_target_hidden(layer: FfnLayerGeometry, budget: CompressionBudgetSpec) -> int:
    if layer.layer_id in budget.per_layer_hidden:
        return int(budget.per_layer_hidden[layer.layer_id])
    target = int(round(layer.hidden_dim * budget.hidden_fraction))
    target = max(target, budget.min_hidden)
    if budget.max_hidden is not None:
        target = min(target, budget.max_hidden)
    target = min(target, layer.hidden_dim)
    return max(target, 1)


def _select_hidden_units(layer: FfnLayerGeometry, budget: CompressionBudgetSpec, k: int) -> torch.Tensor:
    available = torch.arange(layer.hidden_dim)
    live = available[~layer.dead_mask]
    if live.numel() < k:
        live = available
    scores = layer.channel_scores[live]
    if budget.selection_mode == "topk":
        topk = torch.topk(scores, k=k, largest=True).indices
        return live[topk].sort().values
    if budget.selection_mode == "kmeans":
        descriptors = torch.cat(
            [
                _standard_weight(layer.handle.c_fc)[live],
                _standard_weight(layer.handle.c_proj).t()[live],
                layer.mean_hidden_activation[live].unsqueeze(1),
                layer.std_hidden_activation[live].unsqueeze(1),
                layer.mean_hidden_derivative[live].unsqueeze(1),
            ],
            dim=1,
        ).numpy()
        kmeans = KMeans(n_clusters=k, random_state=budget.kmeans_seed, n_init=10)
        labels = kmeans.fit_predict(descriptors)
        chosen: list[int] = []
        for cluster_idx in range(k):
            member_idx = torch.nonzero(torch.tensor(labels == cluster_idx), as_tuple=False).squeeze(1)
            cluster_live = live[member_idx]
            cluster_desc = torch.tensor(descriptors[member_idx], dtype=torch.float32)
            centroid = cluster_desc.mean(dim=0, keepdim=True)
            distances = (cluster_desc - centroid).square().sum(dim=1)
            chosen.append(int(cluster_live[torch.argmin(distances)].item()))
        return torch.tensor(sorted(chosen), dtype=torch.long)
    raise ValueError(f"Unsupported selection_mode {budget.selection_mode!r}")


def _fit_output_projection(
    hidden: torch.Tensor,
    outputs: torch.Tensor,
    *,
    use_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hidden.ndim != 2 or outputs.ndim != 2:
        raise ValueError("hidden and outputs must both be [N, D] matrices.")
    if use_bias:
        design = torch.cat([hidden, torch.ones(hidden.shape[0], 1)], dim=1)
    else:
        design = hidden
    solution = torch.linalg.lstsq(design, outputs).solution
    if use_bias:
        return solution[:-1].t().contiguous(), solution[-1].contiguous()
    return solution.t().contiguous(), None


def derive_ffn_compression(atlas: FfnGeometryAtlas, budget: CompressionBudgetSpec) -> CompressedFfnPack:
    compressed_layers: list[CompressedFfnLayer] = []
    for layer in atlas.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None
        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        student_w_out, student_b_out = _fit_output_projection(hidden, layer.output_sample, use_bias=use_output_bias)
        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj),
            _standard_bias(layer.handle.c_proj),
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out,
            student_b_out,
        )
        compressed_layers.append(
            CompressedFfnLayer(
                layer_id=layer.layer_id,
                activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim,
                compressed_hidden_dim=int(k),
                c_fc_weight=student_w_in,
                c_fc_bias=student_b_in,
                c_proj_weight=student_w_out,
                c_proj_bias=student_b_out,
                selected_indices=selected,
                source_param_count=source_param_count,
                compressed_param_count=compressed_param_count,
                notes={
                    "dead_fraction": layer.stable_stats["dead_fraction"],
                    "selection_mode": budget.selection_mode,
                },
            )
        )
    return CompressedFfnPack(
        method="geometric",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


def _load_compressed_layer(module: CompressedMLP, layer: CompressedFfnLayer) -> None:
    with torch.no_grad():
        module.c_fc.weight.copy_(layer.c_fc_weight)
        if module.c_fc.bias is not None and layer.c_fc_bias is not None:
            module.c_fc.bias.copy_(layer.c_fc_bias)
        module.c_proj.weight.copy_(layer.c_proj_weight)
        if module.c_proj.bias is not None and layer.c_proj_bias is not None:
            module.c_proj.bias.copy_(layer.c_proj_bias)


def _replace_ffn_modules(model: nn.Module, activation_kind: str, pack: CompressedFfnPack) -> nn.Module:
    layer_map = {layer.layer_id: layer for layer in pack.layers}
    handles = _extract_ffn_handles(model, activation_kind)
    for handle in handles:
        if handle.layer_id not in layer_map:
            continue
        layer = layer_map[handle.layer_id]
        new_mlp = CompressedMLP(
            input_dim=layer.c_fc_weight.shape[1],
            hidden_dim=layer.compressed_hidden_dim,
            output_dim=layer.c_proj_weight.shape[0],
            activation_kind=layer.activation_kind,
            c_fc_bias=layer.c_fc_bias is not None,
            c_proj_bias=layer.c_proj_bias is not None,
        )
        _load_compressed_layer(new_mlp, layer)
        new_mlp.to(next(handle.mlp.parameters()).device)
        handle.block.mlp = new_mlp
    return model


def swap_compressed_ffns(
    teacher: TeacherBundle,
    pack: CompressedFfnPack,
    *,
    inplace: bool = False,
) -> nn.Module:
    model = teacher.model if inplace else copy.deepcopy(teacher.model)
    model.eval()
    return _replace_ffn_modules(model, teacher.activation_kind, pack)


def _pack_layer_metrics(
    atlas: FfnGeometryAtlas,
    pack: CompressedFfnPack,
) -> tuple[dict[str, float], ...]:
    atlas_by_id = {layer.layer_id: layer for layer in atlas.layer_geometries}
    metrics: list[dict[str, float]] = []
    for compressed in pack.layers:
        source = atlas_by_id[compressed.layer_id]
        mlp = CompressedMLP(
            input_dim=compressed.c_fc_weight.shape[1],
            hidden_dim=compressed.compressed_hidden_dim,
            output_dim=compressed.c_proj_weight.shape[0],
            activation_kind=compressed.activation_kind,
            c_fc_bias=compressed.c_fc_bias is not None,
            c_proj_bias=compressed.c_proj_bias is not None,
        )
        _load_compressed_layer(mlp, compressed)
        with torch.no_grad():
            predicted = mlp(source.input_sample)
            mse = F.mse_loss(predicted, source.output_sample).item()
            cosine = F.cosine_similarity(
                predicted.reshape(predicted.shape[0], -1),
                source.output_sample.reshape(source.output_sample.shape[0], -1),
                dim=1,
            ).mean().item()
            compressed_preact = source.input_sample @ compressed.c_fc_weight.t()
            if compressed.c_fc_bias is not None:
                compressed_preact = compressed_preact + compressed.c_fc_bias
            jacobian_spectrum = _mean_jacobian_spectrum(
                compressed.c_fc_weight,
                compressed.c_proj_weight,
                compressed_preact,
                compressed.activation_kind,
                jacobian_samples=min(8, source.input_sample.shape[0]),
            )
            denom = source.jacobian_spectrum.norm().item() * jacobian_spectrum.norm().item()
            jacobian_cosine = float(
                torch.dot(source.jacobian_spectrum, jacobian_spectrum).item() / denom
            ) if denom > 0 else 0.0
        metrics.append(
            {
                "layer_id": float(compressed.layer_id),
                "source_hidden_dim": float(compressed.source_hidden_dim),
                "compressed_hidden_dim": float(compressed.compressed_hidden_dim),
                "source_param_count": float(compressed.source_param_count),
                "compressed_param_count": float(compressed.compressed_param_count),
                "layer_output_mse": float(mse),
                "layer_output_cosine": float(cosine),
                "jacobian_spectral_similarity": float(jacobian_cosine),
                "dead_fraction": float(source.stable_stats["dead_fraction"]),
            }
        )
    return tuple(metrics)


def _evaluate_model(
    teacher: TeacherBundle,
    student_model: nn.Module,
    corpus: ProbeCorpus,
    *,
    batch_size: int = 4,
) -> dict[str, float]:
    student_model.eval()
    teacher.model.eval()
    total_tokens = 0
    total_loss = 0.0
    total_kl = 0.0
    total_agreement = 0.0
    with torch.no_grad():
        for input_ids, targets in _iter_lm_batches(corpus.eval_val, batch_size, teacher.device):
            teacher_logits = _forward_logits(teacher.model, input_ids)
            student_logits = _forward_logits(student_model, input_ids)
            loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            )
            teacher_probs = teacher_logits.softmax(dim=-1)
            kl = F.kl_div(
                student_logits.log_softmax(dim=-1),
                teacher_probs,
                reduction="batchmean",
            )
            agreement = (
                teacher_logits.argmax(dim=-1) == student_logits.argmax(dim=-1)
            ).to(dtype=torch.float32).mean()
            tokens = int(targets.numel())
            total_tokens += tokens
            total_loss += float(loss.item()) * tokens
            total_kl += float(kl.item()) * targets.shape[0]
            total_agreement += float(agreement.item()) * targets.shape[0]
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "eval_loss": float(mean_loss),
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
        "teacher_student_kl": float(total_kl / max(corpus.eval_val.shape[0], 1)),
        "top1_agreement": float(total_agreement / max(corpus.eval_val.shape[0], 1)),
        "full_params": float(sum(p.numel() for p in student_model.parameters())),
    }


def run_ffn_geometric_compression(
    teacher: TeacherBundle,
    corpus: ProbeCorpus,
    budget: CompressionBudgetSpec,
    *,
    layer_ids: tuple[int, ...] | None = None,
    batch_size: int = 4,
    max_tokens_per_layer: int = 2048,
) -> CompressionRunResult:
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=layer_ids,
        max_tokens_per_layer=max_tokens_per_layer,
        batch_size=batch_size,
        jacobian_samples=budget.jacobian_samples,
    )
    pack = derive_ffn_compression(atlas, budget)
    student_model = swap_compressed_ffns(teacher, pack, inplace=False)
    metrics = _evaluate_model(teacher, student_model, corpus, batch_size=batch_size)
    metrics["ffn_param_count"] = float(sum(layer.compressed_param_count for layer in pack.layers))
    metrics["source_ffn_param_count"] = float(sum(layer.source_param_count for layer in pack.layers))
    metrics["ffn_compression_ratio"] = float(
        metrics["source_ffn_param_count"] / max(metrics["ffn_param_count"], 1.0)
    )
    return CompressionRunResult(
        method="geometric",
        metrics=metrics,
        layer_metrics=_pack_layer_metrics(atlas, pack),
        pack=pack,
        metadata={"model_name": teacher.model_name},
    )


def _random_pack_like(teacher: TeacherBundle, pack: CompressedFfnPack, seed: int) -> CompressedFfnPack:
    _set_seed(seed)
    random_layers: list[CompressedFfnLayer] = []
    handles = {handle.layer_id: handle for handle in _extract_ffn_handles(teacher.model, teacher.activation_kind)}
    for layer in pack.layers:
        handle = handles[layer.layer_id]
        w_in = _standard_weight(handle.c_fc)
        b_in = _standard_bias(handle.c_fc)
        w_out = _standard_weight(handle.c_proj)
        b_out = _standard_bias(handle.c_proj)
        hidden_dim = layer.compressed_hidden_dim
        scale_in = 1.0 / math.sqrt(w_in.shape[1])
        scale_out = 1.0 / math.sqrt(hidden_dim)
        rand_w_in = torch.empty(hidden_dim, w_in.shape[1]).uniform_(-scale_in, scale_in)
        rand_b_in = None if b_in is None else torch.zeros(hidden_dim)
        rand_w_out = torch.empty(w_out.shape[0], hidden_dim).uniform_(-scale_out, scale_out)
        rand_b_out = None if b_out is None else torch.zeros_like(b_out)
        random_layers.append(
            CompressedFfnLayer(
                layer_id=layer.layer_id,
                activation_kind=layer.activation_kind,
                source_hidden_dim=layer.source_hidden_dim,
                compressed_hidden_dim=hidden_dim,
                c_fc_weight=rand_w_in,
                c_fc_bias=rand_b_in,
                c_proj_weight=rand_w_out,
                c_proj_bias=rand_b_out,
                selected_indices=None,
                source_param_count=layer.source_param_count,
                compressed_param_count=layer.compressed_param_count,
                notes={"baseline": "distillation_random_init"},
            )
        )
    return CompressedFfnPack(
        method="distilled",
        model_family=pack.model_family,
        dataset_name=pack.dataset_name,
        layers=tuple(random_layers),
        metadata=pack.metadata,
    )


def _parameters_for_target_ffns(model: nn.Module, layer_ids: set[int], activation_kind: str) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for handle in _extract_ffn_handles(model, activation_kind):
        if handle.layer_id in layer_ids:
            params.extend(list(handle.mlp.parameters()))
    return params


def run_ffn_distillation_baseline(spec: DistillationSpec) -> CompressionRunResult:
    _set_seed(spec.seed)
    layer_ids = spec.layer_ids or spec.teacher.layer_ids
    atlas = probe_ffn_geometry(
        spec.teacher,
        spec.corpus,
        layer_ids=layer_ids,
        max_tokens_per_layer=2048,
        batch_size=spec.batch_size,
        jacobian_samples=spec.budget.jacobian_samples,
    )
    geometric_pack = derive_ffn_compression(atlas, spec.budget)
    distilled_pack = _random_pack_like(spec.teacher, geometric_pack, seed=spec.seed)
    student_model = swap_compressed_ffns(spec.teacher, distilled_pack, inplace=False).to(spec.teacher.device)
    for param in student_model.parameters():
        param.requires_grad = False
    trainable = _parameters_for_target_ffns(student_model, set(layer_ids), spec.teacher.activation_kind)
    for param in trainable:
        param.requires_grad = True
    optimizer = torch.optim.Adam(trainable, lr=spec.lr)

    teacher_handles = [handle for handle in _extract_ffn_handles(spec.teacher.model, spec.teacher.activation_kind) if handle.layer_id in set(layer_ids)]
    student_handles = [handle for handle in _extract_ffn_handles(student_model, spec.teacher.activation_kind) if handle.layer_id in set(layer_ids)]

    step = 0
    while step < spec.steps:
        for input_ids, targets in _iter_lm_batches(spec.corpus.probe_train, spec.batch_size, spec.teacher.device):
            optimizer.zero_grad()
            with _collect_ffn_outputs(teacher_handles) as teacher_outputs, _collect_ffn_outputs(student_handles) as student_outputs:
                with torch.no_grad():
                    teacher_logits = _forward_logits(spec.teacher.model, input_ids)
                student_logits = _forward_logits(student_model, input_ids)
            kl = F.kl_div(
                student_logits.log_softmax(dim=-1),
                teacher_logits.softmax(dim=-1),
                reduction="batchmean",
            )
            ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                targets.reshape(-1),
            )
            layer_losses = []
            for layer_id in teacher_outputs:
                if not teacher_outputs[layer_id] or not student_outputs[layer_id]:
                    continue
                teacher_block = torch.cat(teacher_outputs[layer_id], dim=0).to(spec.teacher.device)
                student_block = torch.cat(student_outputs[layer_id], dim=0).to(spec.teacher.device)
                layer_losses.append(F.mse_loss(student_block, teacher_block))
            layer_loss = torch.stack(layer_losses).mean() if layer_losses else torch.tensor(0.0, device=spec.teacher.device)
            loss = spec.kl_weight * kl + spec.target_ce_weight * ce + spec.layer_output_weight * layer_loss
            loss.backward()
            optimizer.step()
            step += 1
            if step >= spec.steps:
                break

    metrics = _evaluate_model(spec.teacher, student_model, spec.corpus, batch_size=spec.batch_size)
    metrics["ffn_param_count"] = float(sum(layer.compressed_param_count for layer in distilled_pack.layers))
    metrics["source_ffn_param_count"] = float(sum(layer.source_param_count for layer in distilled_pack.layers))
    metrics["ffn_compression_ratio"] = float(
        metrics["source_ffn_param_count"] / max(metrics["ffn_param_count"], 1.0)
    )
    return CompressionRunResult(
        method="distilled",
        metrics=metrics,
        layer_metrics=_pack_layer_metrics(atlas, distilled_pack),
        pack=distilled_pack,
        metadata={"steps": spec.steps},
    )


def _neuron_descriptors(layer: FfnLayerGeometry) -> torch.Tensor:
    source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
    source_w_out = _standard_weight(layer.handle.c_proj).t().to(dtype=torch.float32)
    bias = _standard_bias(layer.handle.c_fc)
    if bias is None:
        bias = torch.zeros(layer.hidden_dim, dtype=torch.float32)
    descriptors = torch.cat(
        [
            F.normalize(source_w_in, dim=1),
            F.normalize(source_w_out, dim=1),
            bias.unsqueeze(1).to(dtype=torch.float32),
            layer.mean_hidden_activation.unsqueeze(1),
            layer.std_hidden_activation.unsqueeze(1),
            layer.mean_hidden_derivative.unsqueeze(1),
            layer.channel_scores.unsqueeze(1),
        ],
        dim=1,
    )
    return descriptors


def run_ffn_learned_compressor(spec: LearnedCompressorSpec) -> CompressionRunResult:
    _set_seed(spec.seed)
    layer_ids = spec.layer_ids or spec.teacher.layer_ids
    atlas = probe_ffn_geometry(
        spec.teacher,
        spec.corpus,
        layer_ids=layer_ids,
        max_tokens_per_layer=2048,
        batch_size=4,
        jacobian_samples=spec.budget.jacobian_samples,
    )
    descriptors = []
    layer_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for layer in atlas.layer_geometries:
        desc = _neuron_descriptors(layer)
        descriptors.append(desc)
        layer_offsets[layer.layer_id] = (cursor, cursor + desc.shape[0])
        cursor += desc.shape[0]
    descriptor_matrix = torch.cat(descriptors, dim=0).to(spec.teacher.device)

    autoencoder = DescriptorAutoencoder(
        input_dim=descriptor_matrix.shape[1],
        hidden_dim=spec.autoencoder_hidden,
        code_dim=spec.code_dim,
    ).to(spec.teacher.device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=spec.lr)

    dataset = descriptor_matrix[torch.randperm(descriptor_matrix.shape[0])]
    for _epoch in range(spec.epochs):
        for start in range(0, dataset.shape[0], spec.batch_size):
            batch = dataset[start : start + spec.batch_size]
            optimizer.zero_grad()
            _code, recon = autoencoder(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        all_codes, _ = autoencoder(descriptor_matrix)
    learned_layers: list[CompressedFfnLayer] = []
    for layer in atlas.layer_geometries:
        start, end = layer_offsets[layer.layer_id]
        codes = all_codes[start:end].detach().cpu().numpy()
        k = _layer_target_hidden(layer, spec.budget)
        kmeans = KMeans(n_clusters=k, random_state=spec.seed, n_init=10)
        labels = kmeans.fit_predict(codes)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=spec.teacher.device)
        decoded = autoencoder.decoder(centroids).detach().cpu()
        input_dim = layer.input_dim
        output_dim = layer.input_dim
        source_bias = _standard_bias(layer.handle.c_fc)
        use_proj_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = decoded[:, :input_dim].contiguous()
        if source_bias is None:
            student_b_in = None
        else:
            student_b_in = decoded[:, 2 * input_dim].contiguous()

        fc_output = layer.input_sample @ student_w_in.t()
        if student_b_in is not None:
            fc_output = fc_output + student_b_in
        student_hidden = _apply_activation(fc_output, layer.activation_kind)
        student_w_out, student_b_out = _fit_output_projection(student_hidden, layer.output_sample, use_bias=use_proj_bias)

        source_param_count = _param_count(_standard_weight(layer.handle.c_fc), _standard_bias(layer.handle.c_fc))
        source_param_count += _param_count(_standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        compressed_param_count = _param_count(student_w_in, student_b_in)
        compressed_param_count += _param_count(student_w_out, student_b_out)
        learned_layers.append(
            CompressedFfnLayer(
                layer_id=layer.layer_id,
                activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim,
                compressed_hidden_dim=k,
                c_fc_weight=student_w_in,
                c_fc_bias=student_b_in,
                c_proj_weight=student_w_out,
                c_proj_bias=student_b_out,
                selected_indices=None,
                source_param_count=source_param_count,
                compressed_param_count=compressed_param_count,
                notes={"compressor": "descriptor_autoencoder"},
            )
        )

    pack = CompressedFfnPack(
        method="learned_autoencoder",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(learned_layers),
        metadata={"code_dim": spec.code_dim, "epochs": spec.epochs},
    )
    student_model = swap_compressed_ffns(spec.teacher, pack, inplace=False)
    metrics = _evaluate_model(spec.teacher, student_model, spec.corpus, batch_size=4)
    metrics["ffn_param_count"] = float(sum(layer.compressed_param_count for layer in pack.layers))
    metrics["source_ffn_param_count"] = float(sum(layer.source_param_count for layer in pack.layers))
    metrics["ffn_compression_ratio"] = float(
        metrics["source_ffn_param_count"] / max(metrics["ffn_param_count"], 1.0)
    )
    return CompressionRunResult(
        method="learned_autoencoder",
        metrics=metrics,
        layer_metrics=_pack_layer_metrics(atlas, pack),
        pack=pack,
        metadata={"code_dim": spec.code_dim},
    )


def result_to_dict(result: CompressionRunResult) -> dict[str, Any]:
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        return str(obj)

    return _serialize(result)
