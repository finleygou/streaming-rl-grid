"""Two-group sparse tile coding for position and goal-relative features."""

from hashlib import blake2b
from math import floor
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import AgentConfig, EnvironmentConfig


class IndexHashTable:
    """Exact indices until full, then deterministic hashing with collision counts."""

    def __init__(self, size: int):
        self.size = int(size)
        self.dictionary: Dict[Tuple[Any, ...], int] = {}
        self.overfull_count = 0

    def get_index(self, key: Iterable[Any], readonly: bool = False) -> int:
        key_tuple = tuple(key)
        if key_tuple in self.dictionary:
            return self.dictionary[key_tuple]
        if readonly:
            return -1
        if len(self.dictionary) < self.size:
            index = len(self.dictionary)
            self.dictionary[key_tuple] = index
            return index
        self.overfull_count += 1
        payload = repr(key_tuple).encode("utf-8")
        return int.from_bytes(blake2b(payload, digest_size=8).digest(), "little") % self.size

    def state_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "dictionary": self.dictionary.copy(),
            "overfull_count": self.overfull_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("Checkpoint IHT size does not match the configured size.")
        self.dictionary = {tuple(k): int(v) for k, v in state["dictionary"].items()}
        self.overfull_count = int(state["overfull_count"])


def tiles(
    iht: IndexHashTable,
    num_tilings: int,
    floats: Sequence[float],
    ints: Sequence[int],
    readonly: bool = False,
) -> List[int]:
    """Sutton's asymmetric-offset tiles3 construction."""

    quantized = [floor(value * num_tilings) for value in floats]
    active: List[int] = []
    for tiling in range(num_tilings):
        offset = tiling
        coordinates: List[int] = [tiling]
        for value in quantized:
            coordinates.append((value + offset) // num_tilings)
            offset += 2 * tiling
        index = iht.get_index(coordinates + list(ints), readonly=readonly)
        if index >= 0:
            active.append(index)
    return active


class DualTileCoder:
    """Position and relative-goal tile groups plus one categorical bias feature."""

    def __init__(self, env_config: EnvironmentConfig, agent_config: AgentConfig):
        self.width = env_config.width
        self.height = env_config.height
        self.num_tilings = agent_config.num_tilings
        self.tiles_per_dimension = agent_config.tiles_per_dimension
        self.iht = IndexHashTable(agent_config.iht_size)

    @property
    def nominal_active_count(self) -> int:
        return 2 * self.num_tilings + 1

    @property
    def size(self) -> int:
        return self.iht.size

    def active(self, observation: Sequence[int], action: int, readonly: bool = False) -> np.ndarray:
        x, y, gx, gy, previous_action = [int(v) for v in observation]
        pos = [
            self._scale(x, self.width),
            self._scale(y, self.height),
        ]
        relative = [
            self._scale_signed(gx - x, self.width - 1),
            self._scale_signed(gy - y, self.height - 1),
        ]
        first = tiles(self.iht, self.num_tilings, pos, [0, previous_action, int(action)], readonly=readonly)
        second = tiles(self.iht, self.num_tilings, relative, [1, previous_action, int(action)], readonly=readonly)
        bias = self.iht.get_index(("bias", previous_action, int(action)), readonly=readonly)
        if bias >= 0:
            first.append(bias)
        # Collisions after the IHT fills must not turn a binary feature into a count feature.
        return np.unique(np.asarray(first + second, dtype=np.int64))

    def _scale(self, value: int, size: int) -> float:
        denominator = max(1, size - 1)
        return (float(value) / denominator) * self.tiles_per_dimension

    def _scale_signed(self, value: int, maximum_abs: int) -> float:
        denominator = max(1, maximum_abs)
        normalized = (float(value) / denominator + 1.0) * 0.5
        return normalized * self.tiles_per_dimension

    def state_dict(self) -> Dict[str, Any]:
        return {"iht": self.iht.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.iht.load_state_dict(state["iht"])
