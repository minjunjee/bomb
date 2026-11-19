"""submission.py sources for the simulated student networks.

Every student network gets ITS OWN self-contained submission.py, written exactly as a student would submit it:
the notebook's submission cell (`%%writefile submission.py`, cell 11 of AIC20000_bomb_competition_2026.ipynb) with
only the `Net` class edited. "base" is the notebook's default Net, untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK = ROOT / "AIC20000_bomb_competition_2026.ipynb"

# "a slightly bigger network (residual blocks)", the way a student would write it after reading "Things to try".
RES = '''
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_CHANNELS, 64, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 9 * 9, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))
'''

# another student's version: narrower, two blocks, layers written out one by one
RES2 = '''
class Block(nn.Module):
    """conv - relu - conv, plus the input (a residual block)"""

    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.b = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.a(x))
        y = self.b(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(IN_CHANNELS, 48, kernel_size=3, padding=1)
        self.res1 = Block(48)
        self.res2 = Block(48)
        self.fc1 = nn.Linear(48 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.res2(self.res1(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)
'''

ARCHS = {"base": None, "res": RES, "res2": RES2}


def notebook_submission_cell() -> str:
    """The text `%%writefile submission.py` saves: the notebook's submission cell without the magic line."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][11]["source"])
    first, rest = src.split("\n", 1)
    assert first.strip() == "%%writefile submission.py", first
    return rest if rest.endswith("\n") else rest + "\n"


def submission_source(arch: str = "base") -> str:
    src = notebook_submission_cell()
    if ARCHS[arch] is None:
        return src
    head, rest = src.split("class Net(nn.Module):", 1)
    _, tail = rest.split("        return self.layers(x)\n", 1)
    return head + ARCHS[arch].strip("\n") + "\n" + tail


def write_submission_py(folder: str | Path, arch: str = "base") -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "submission.py"
    path.write_text(submission_source(arch), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    print(submission_source(sys.argv[1] if len(sys.argv) > 1 else "base"))
