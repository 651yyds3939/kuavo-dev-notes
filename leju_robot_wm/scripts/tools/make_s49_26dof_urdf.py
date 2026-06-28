#!/usr/bin/env python3
"""Build S49 26-DoF URDF for RL training (leg12 + arm14, no fingers/head)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Finger + head revolute joints to freeze for RL (match S46 USD DoF count).
FREEZE_JOINTS = {
    "l_thumbCMC",
    "l_thumbMCP",
    "l_indexMCP",
    "l_indexPIP",
    "l_middleMCP",
    "l_middlePIP",
    "l_ringMCP",
    "l_ringPIP",
    "l_littleMCP",
    "l_littlePIP",
    "r_thumbCMC",
    "r_thumbMCP",
    "r_indexMCP",
    "r_indexPIP",
    "r_middleMCP",
    "r_middlePIP",
    "r_ringMCP",
    "r_ringPIP",
    "r_littleMCP",
    "r_littlePIP",
    "zhead_1_joint",
    "zhead_2_joint",
}

JOINT_REVOLUTE_RE = re.compile(r'<joint name="(?P<name>[^"]+)" type="revolute">')


def make_26dof_urdf(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    frozen = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal frozen
        name = match.group("name")
        if name in FREEZE_JOINTS:
            frozen += 1
            return f'<joint name="{name}" type="fixed">'
        return match.group(0)

    text = JOINT_REVOLUTE_RE.sub(_replace, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"Wrote {dst} (frozen {frozen} extra revolute joints -> fixed)")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.urdf> <output_26dof.urdf>", file=sys.stderr)
        sys.exit(1)
    make_26dof_urdf(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
