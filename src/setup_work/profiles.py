"""Profile resolution: `extends` composition into one predictable setup plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config, ProfileSpec, deep_merge
from .errors import ConfigError, NotFoundError


def _union(*lists: list[str]) -> list[str]:
    out: list[str] = []
    for lst in lists:
        for x in lst:
            if x not in out:
                out.append(x)
    return out


@dataclass
class ResolvedProfile:
    name: str
    chain: list[str]
    description: str
    tools_required: list[str] = field(default_factory=list)
    tools_optional: list[str] = field(default_factory=list)
    harnesses_required: list[str] = field(default_factory=list)
    harnesses_optional: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)
    model_preferences: dict[str, Any] = field(default_factory=dict)
    model_pull: list[str] = field(default_factory=list)
    inference: dict[str, Any] = field(default_factory=dict)
    validation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def resolve_profile(cfg: Config, name: str) -> ResolvedProfile:
    profiles = cfg.profiles.profiles
    if name not in profiles:
        raise NotFoundError(f"unknown profile {name!r}", hint=f"profiles: {', '.join(profiles)}")

    order: list[str] = []

    def visit(n: str, stack: tuple[str, ...]) -> None:
        if n in stack:
            raise ConfigError(f"profile cycle: {' -> '.join((*stack, n))}")
        if n not in profiles:
            raise ConfigError(f"profile {stack[-1]} extends unknown profile {n!r}")
        for parent in profiles[n].extends:
            visit(parent, (*stack, n))
        if n not in order:
            order.append(n)

    visit(name, ())
    out = ResolvedProfile(name, order, profiles[name].description)
    for n in order:  # parents first, so children override mappings
        p: ProfileSpec = profiles[n]
        out.tools_required = _union(out.tools_required, p.tools.required)
        out.tools_optional = _union(out.tools_optional, p.tools.optional)
        out.harnesses_required = _union(out.harnesses_required, p.harnesses.required)
        out.harnesses_optional = _union(out.harnesses_optional, p.harnesses.optional)
        out.engines = _union(out.engines, p.engines)
        out.model_preferences = deep_merge(out.model_preferences, p.models.preferences)
        out.model_pull = _union(out.model_pull, p.models.pull)
        out.inference = deep_merge(out.inference, p.inference)
        out.validation = _union(out.validation, p.validation)
    out.tools_optional = [t for t in out.tools_optional if t not in out.tools_required]
    out.harnesses_optional = [h for h in out.harnesses_optional if h not in out.harnesses_required]
    return out
