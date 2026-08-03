"""Shared input and output contracts for independent research skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillInput:
    user_query: str
    parsed_entities: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    condition_evidence: list[dict[str, Any]]
    formula_records: list[dict[str, Any]]
    support_evidence: list[dict[str, Any]]
    counter_evidence: list[dict[str, Any]]
    condition_dependent_evidence: list[dict[str, Any]]
    dataset_version: str
    previous_output: "SkillOutput | None" = None
    evidence_bundle: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    skill_name: str
    direct_answer: str
    structured_reasoning: str
    uncertainty: str
    evidence_cards: list[dict[str, Any]]
    quality_gate: dict[str, Any]
    missing_evidence: list[str]
    specific_fields: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "direct_answer": self.direct_answer,
            "structured_reasoning": self.structured_reasoning,
            "uncertainty": self.uncertainty,
            "evidence_cards": self.evidence_cards,
            "quality_gate": self.quality_gate,
            "missing_evidence": self.missing_evidence,
            "specific_fields": self.specific_fields,
            "trace": self.trace,
        }
