"""Prompt-only target key/value caching for an approximate CTMC target process.

This is a Fast-dLLM-inspired adaptation. It always caches exactly the original
prompt positions, recomputes every generated-suffix position, and refreshes the
prompt snapshot after a configured number of committed reveals.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class PromptOnlyTargetCacheModel(torch.nn.Module):
    """Expose cached suffix logits through the existing full-state model contract."""

    prompt_cache_enabled = True
    cache_mode = "prompt_only_target_kv"
    preserved_sequential_distribution = "prompt_cached_sequential_ctmc"

    def __init__(self, inner_model: Any, refresh_interval: int):
        super().__init__()
        if int(refresh_interval) not in (8, 16, 32):
            raise ValueError("prompt cache refresh interval must be 8, 16 or 32")
        self.inner_model = inner_model
        self.prompt_cache_refresh_interval = int(refresh_interval)
        self.prompt_length: Optional[int] = None
        self.target_batch_size: Optional[int] = None
        self._prompt_past_key_values = None
        self._refresh_required = True
        self._committed_since_refresh = 0
        self._total_committed = 0
        self._refresh_epoch = 0
        self._stats: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []

    @property
    def device(self) -> torch.device:
        device = getattr(self.inner_model, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.inner_model.parameters()).device

    @property
    def config(self):
        return self.inner_model.config

    def get_output_embeddings(self):
        return self.inner_model.get_output_embeddings()

    def begin_request(self, prompt_length: int, target_batch_size: int) -> None:
        if prompt_length <= 0:
            raise ValueError("prompt-only caching requires a nonempty prompt")
        if target_batch_size <= 0:
            raise ValueError("target batch size must be positive")
        self.prompt_length = int(prompt_length)
        self.target_batch_size = int(target_batch_size)
        self._prompt_past_key_values = None
        self._refresh_required = True
        self._committed_since_refresh = 0
        self._total_committed = 0
        self._refresh_epoch = 0
        self._stats = {
            "prompt_cache_refreshes": 0,
            "prompt_cache_full_input_rows": 0,
            "prompt_cache_suffix_input_rows": 0,
            "prompt_cache_suffix_calls": 0,
            "prompt_cache_discarded_tentative_updates": 0,
            "prompt_cache_terminal_forced_refreshes": 0,
            "prompt_cache_construction_seconds": 0.0,
            "prompt_cache_periodic_refresh_seconds": 0.0,
            "prompt_cache_total_refresh_seconds": 0.0,
            "prompt_cache_copy_seconds": 0.0,
            "prompt_cache_suffix_forward_seconds": 0.0,
        }
        self._events = []

    def remaining_before_refresh(self) -> int:
        if self.prompt_length is None:
            raise RuntimeError("prompt cache request has not started")
        if self._refresh_required:
            return self.prompt_cache_refresh_interval
        return self.prompt_cache_refresh_interval - self._committed_since_refresh

    def note_commits(self, count: int) -> None:
        count = int(count)
        if count <= 0:
            raise ValueError("committed reveal count must be positive")
        remaining = self.remaining_before_refresh()
        if count > remaining:
            raise RuntimeError("a speculative round crossed a prompt-cache refresh boundary")
        self._committed_since_refresh += count
        self._total_committed += count
        if self._committed_since_refresh == self.prompt_cache_refresh_interval:
            self._refresh_required = True
            self._events.append(
                {
                    "kind": "refresh_boundary_reached",
                    "total_committed_reveals": self._total_committed,
                    "refresh_epoch": self._refresh_epoch,
                }
            )

    def force_refresh_before_terminal_cleanup(self) -> None:
        if self.prompt_length is None:
            raise RuntimeError("prompt cache request has not started")
        if not self._refresh_required:
            self._stats["prompt_cache_terminal_forced_refreshes"] += 1
        self._refresh_required = True

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _state_digest(state: torch.Tensor) -> str:
        payload = state.detach().to("cpu").contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def _refresh(self, full_states: torch.Tensor) -> None:
        if self.prompt_length is None or self.target_batch_size is None:
            raise RuntimeError("prompt cache request has not started")
        if full_states.ndim != 2 or full_states.shape[0] != self.target_batch_size:
            raise ValueError("prompt refresh requires the fixed target batch shape")
        reference = full_states[:1]
        refresh_states = reference.expand(self.target_batch_size, -1)
        device = full_states.device
        self._sync(device)
        refresh_started = time.perf_counter()
        output = self.inner_model(refresh_states, use_cache=True)
        self._sync(device)
        refresh_model_seconds = time.perf_counter() - refresh_started
        past = getattr(output, "past_key_values", None)
        if past is None:
            raise RuntimeError("cache-capable model did not return past_key_values")
        self._sync(device)
        copy_started = time.perf_counter()
        prompt_cache = []
        for layer in past:
            if len(layer) != 2:
                raise RuntimeError("every cache layer must contain key and value tensors")
            key, value = layer
            if key.shape[0] != self.target_batch_size or value.shape[0] != self.target_batch_size:
                raise RuntimeError("cache batch shape differs from fixed target batch size")
            if key.shape[-2] < self.prompt_length or value.shape[-2] < self.prompt_length:
                raise RuntimeError("cache sequence is shorter than the prompt")
            prompt_cache.append(
                (
                    key[:, :, : self.prompt_length].detach().clone(),
                    value[:, :, : self.prompt_length].detach().clone(),
                )
            )
        self._sync(device)
        copy_seconds = time.perf_counter() - copy_started
        refresh_seconds = refresh_model_seconds + copy_seconds
        initial_construction = self._refresh_epoch == 0
        self._prompt_past_key_values = tuple(prompt_cache)
        self._refresh_required = False
        self._committed_since_refresh = 0
        self._refresh_epoch += 1
        self._stats["prompt_cache_refreshes"] += 1
        self._stats["prompt_cache_full_input_rows"] += self.target_batch_size
        self._stats["prompt_cache_total_refresh_seconds"] += refresh_seconds
        self._stats["prompt_cache_copy_seconds"] += copy_seconds
        if initial_construction:
            self._stats["prompt_cache_construction_seconds"] += refresh_seconds
        else:
            self._stats["prompt_cache_periodic_refresh_seconds"] += refresh_seconds
        self._events.append(
            {
                "kind": "prompt_cache_refresh",
                "refresh_epoch": self._refresh_epoch,
                "total_committed_reveals": self._total_committed,
                "state_sha256": self._state_digest(reference),
                "prompt_length": self.prompt_length,
                "target_batch_size": self.target_batch_size,
                "refresh_model_seconds": refresh_model_seconds,
                "copy_seconds": copy_seconds,
                "refresh_seconds": refresh_seconds,
                "initial_construction": initial_construction,
            }
        )

    def forward(self, input_ids: torch.Tensor, num_logits_to_keep: int = 0):
        if self.prompt_length is None or self.target_batch_size is None:
            raise RuntimeError("prompt cache request has not started")
        if input_ids.ndim != 2 or input_ids.shape[0] != self.target_batch_size:
            raise ValueError("cached target evaluation requires the fixed target batch shape")
        if input_ids.shape[1] <= self.prompt_length:
            raise ValueError("cached target evaluation requires a nonempty suffix")
        if self._refresh_required or self._prompt_past_key_values is None:
            self._refresh(input_ids)
        suffix = input_ids[:, self.prompt_length :]
        self._sync(input_ids.device)
        suffix_started = time.perf_counter()
        output = self.inner_model(
            suffix,
            past_key_values=self._prompt_past_key_values,
            use_cache=False,
        )
        self._sync(input_ids.device)
        suffix_seconds = time.perf_counter() - suffix_started
        logits = output.logits
        if logits.shape[:2] != suffix.shape:
            raise RuntimeError("cached model logits do not align with the complete generated suffix")
        if num_logits_to_keep and num_logits_to_keep != suffix.shape[1]:
            raise ValueError("prompt cache wrapper only exposes the complete generated suffix")
        self._stats["prompt_cache_suffix_calls"] += 1
        self._stats["prompt_cache_suffix_input_rows"] += self.target_batch_size
        self._stats["prompt_cache_suffix_forward_seconds"] += suffix_seconds
        return output

    def metrics(self) -> Dict[str, Any]:
        return dict(self._stats)

    def trace(self) -> Dict[str, Any]:
        return {
            "cache_mode": self.cache_mode,
            "refresh_interval": self.prompt_cache_refresh_interval,
            "refresh_count": self._stats.get("prompt_cache_refreshes", 0),
            "total_committed_reveals": self._total_committed,
            "committed_since_refresh": self._committed_since_refresh,
            "refresh_required": self._refresh_required,
            "events": list(self._events),
            "persistent_cache_updates_from_tentative_states": 0,
            "preserved_sequential_distribution": self.preserved_sequential_distribution,
            "full_forward_reference_distribution": "released_checkpoint_full_forward_target",
        }
