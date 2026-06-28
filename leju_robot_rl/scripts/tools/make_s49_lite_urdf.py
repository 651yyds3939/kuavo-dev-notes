#!/usr/bin/env python3
"""Strip non-essential collision geometry from S49 26-DoF URDF for Isaac training.

S46 uses a pre-baked USD with simplified collisions. Runtime S49 URDF still carries
finger/camera/head collision meshes, which slows PhysX and makes standing harder.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Link name substrings whose collision elements are removed (meshes stay for visuals).
STRIP_COLLISION_SUBSTRINGS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "palm",
    "camera",
    "radar",
    "head",
    "zhead",
    "dummy",
    "tripod",
    "end_effect",
    "hand",
)

LINK_OPEN_RE = re.compile(r'<link name="(?P<name>[^"]+)">')
LINK_CLOSE = "</link>"


def _should_strip_collision(link_name: str) -> bool:
    lower = link_name.lower()
    return any(token in lower for token in STRIP_COLLISION_SUBSTRINGS)


def _strip_link_collisions(link_block: str, link_name: str) -> str:
    if not _should_strip_collision(link_name):
        return link_block
    return re.sub(r"\s*<collision>.*?</collision>\s*", "\n", link_block, flags=re.DOTALL)


def make_lite_urdf(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    out_parts: list[str] = []
    pos = 0
    stripped_links = 0
    stripped_collisions = 0

    for match in LINK_OPEN_RE.finditer(text):
        out_parts.append(text[pos : match.start()])
        link_name = match.group("name")
        link_start = match.start()
        close_idx = text.find(LINK_CLOSE, match.end())
        if close_idx < 0:
            raise ValueError(f"Unclosed link block: {link_name}")
        link_end = close_idx + len(LINK_CLOSE)
        link_block = text[link_start:link_end]
        if _should_strip_collision(link_name):
            before = link_block.count("<collision>")
            link_block = _strip_link_collisions(link_block, link_name)
            after = link_block.count("<collision>")
            if before > after:
                stripped_links += 1
                stripped_collisions += before - after
        out_parts.append(link_block)
        pos = link_end

    out_parts.append(text[pos:])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(out_parts), encoding="utf-8")
    print(
        f"Wrote {dst} (stripped collisions on {stripped_links} links, "
        f"removed {stripped_collisions} collision elements)"
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_26dof.urdf> <output_lite.urdf>", file=sys.stderr)
        sys.exit(1)
    make_lite_urdf(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
