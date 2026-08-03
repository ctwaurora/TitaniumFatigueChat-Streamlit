"""Explicit routing and traceable chaining for four independent skills."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

from src.research_skills import experiment_design_skill
from src.research_skills import hypothesis_generation_skill
from src.research_skills import research_gap_skill
from src.research_skills import scientific_analysis_skill
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_MODULES: dict[str, ModuleType] = {
    "research_analysis": scientific_analysis_skill,
    "scientific_analysis": scientific_analysis_skill,
    "scientific_analysis_skill": scientific_analysis_skill,
    "research_gap": research_gap_skill,
    "research_gap_skill": research_gap_skill,
    "hypothesis_generation": hypothesis_generation_skill,
    "hypothesis_generation_skill": hypothesis_generation_skill,
    "experiment_design": experiment_design_skill,
    "experiment_design_skill": experiment_design_skill,
}


def get_research_skill(mode: str) -> ModuleType:
    """Return a concrete skill module; legacy smart-search maps to analysis."""
    normalized = "research_analysis" if mode == "smart_search" else str(mode)
    if normalized not in SKILL_MODULES:
        raise ValueError(f"Unsupported research skill mode: {mode}")
    return SKILL_MODULES[normalized]


def run_skill_chain(value: SkillInput) -> list[SkillOutput]:
    """Run all skills in order while preserving the original evidence input."""
    outputs = []
    current = value
    for skill in (
        scientific_analysis_skill,
        research_gap_skill,
        hypothesis_generation_skill,
        experiment_design_skill,
    ):
        output = skill.generate(current)
        outputs.append(output)
        # Only structured output is chained. All original evidence fields stay
        # unchanged and every later skill validates them again.
        current = replace(value, previous_output=output)
    return outputs
