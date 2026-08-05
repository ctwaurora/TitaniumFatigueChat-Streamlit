"""In-memory stage telemetry for one research request."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


STAGES = ("PLANNING", "RETRIEVING", "VERIFYING", "REASONING", "PUBLISHING", "COMPLETED", "FAILED")


@dataclass
class SkillRunTelemetry:
    started: float = field(default_factory=time.perf_counter)
    current_stage: str = "PLANNING"
    stage_started: float = field(default_factory=time.perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    transitions: list[str] = field(default_factory=lambda: ["PLANNING"])
    counters: dict[str, int] = field(default_factory=dict)
    failure_reason: str = ""

    def transition(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unsupported telemetry stage: {stage}")
        now = time.perf_counter()
        self.stage_seconds[self.current_stage] = round(
            self.stage_seconds.get(self.current_stage, 0.0) + now - self.stage_started, 6
        )
        self.current_stage = stage
        self.stage_started = now
        self.transitions.append(stage)

    def count(self, **values: int) -> None:
        for key, value in values.items():
            self.counters[key] = int(value)

    def fail(self, reason: str) -> None:
        self.failure_reason = str(reason)[:240]
        self.transition("FAILED")

    def as_dict(self) -> dict[str, Any]:
        now = time.perf_counter()
        durations = dict(self.stage_seconds)
        durations[self.current_stage] = round(
            durations.get(self.current_stage, 0.0) + now - self.stage_started, 6
        )
        return {
            "current_stage": self.current_stage,
            "transitions": list(self.transitions),
            "stage_seconds": durations,
            "total_seconds": round(now - self.started, 6),
            "counters": dict(self.counters),
            "failure_reason": self.failure_reason,
        }
