"""Draw a game as text or as an image.
Board sprites: CoderOne Bomberland (MIT, assets/LICENSE-bomberland.md). Characters: tools/draw_mascots.py."""

from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .game import Game

ASSETS = Path(__file__).parent / "assets"


def to_text(game: Game) -> str:
    grid = [["#" if w else "." for w in row] for row in game.walls]
    for r, c in game.last_blast:
        grid[r][c] = "*"
    for (r, c), bomb in game.bombs.items():
        grid[r][c] = str(min(bomb.fuse, 9))
    for p, (r, c) in enumerate(game.pos):
        grid[r][c] = "AB"[p]
    head = f"t={game.t}  lives {game.lives[0]}:{game.lives[1]}  kills {game.kills[0]}:{game.kills[1]}"
    return "\n".join([head] + ["".join(row) for row in grid])


@cache
def _sprites(tile: int) -> dict[str, Image.Image]:
    return {f.stem: Image.open(f).convert("RGBA").resize((tile, tile), Image.NEAREST) for f in ASSETS.glob("*.png")}


def to_rgb(game: Game, tile: int = 64) -> np.ndarray:
    """(H, W, 3) image: a status bar above the board."""
    sprites, bar = _sprites(tile), tile // 2
    img = Image.new("RGB", (game.size * tile, bar + game.size * tile), (18, 22, 29))
    draw = ImageDraw.Draw(img)

    def put(name, r, c):
        img.paste(sprites[name], (c * tile, bar + r * tile), sprites[name])

    for r in range(game.size):
        for c in range(game.size):
            put(f"floor_{(r * 7 + c * 3) % 5 + 1}", r, c)
            if game.walls[r, c]:
                put("wall", r, c)
            if (r, c) in game.last_blast:
                put("blast", r, c)
    for (r, c), bomb in game.bombs.items():
        put("bomb_" + "ab"[bomb.owner], r, c)
        draw.text((c * tile + tile - 4, bar + r * tile + 2), str(bomb.fuse), fill="white", anchor="ra", font_size=tile // 3)
    for p, (r, c) in enumerate(game.pos):
        put("player_" + "ab"[p], r, c)
    status = f"A  lives {game.lives[0]}  kills {game.kills[0]}     t {game.t}/{game.rules.max_steps}     kills {game.kills[1]}  lives {game.lives[1]}  B"
    draw.text((img.width // 2, bar // 2), status, fill="white", anchor="mm", font_size=bar * 2 // 3)
    return np.asarray(img)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 4):
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
