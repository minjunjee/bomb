"""Opponent panels: who the network trains against and who it is graded against locally.

TRAIN  opponents the network may train against (RL league, data generation)
VAL    held-out bots, used ONLY to choose between experiments (never trained against)
TEST   held-out bots, used ONLY for the final report (never used for any decision)

A spec that does not exist yet (file missing) is skipped by `available()`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASELINES = ["random", "simple", "zoo.base_cautious"]

TRAIN_RULE = ["zoo.ta_smart_a", "zoo.ta_smart_b", "zoo.evader", "zoo.ta_final_a", "zoo.ta_final_b"]
TRAIN_PARAM = [f"zoo.param:{s}" for s in range(0, 200)]  # seeds 0..199 of the parametrised family
EXPERTS = ["zoo.x_solver", "zoo.x_matrix", "zoo.x_exploit", "zoo.c_expert"]
TRAIN = BASELINES + TRAIN_RULE + EXPERTS

VAL_PARAM = [f"zoo.param:{s}" for s in range(9000, 9006)]  # seeds never used for training
VAL = ["zoo.val_smart", "zoo.val_hunter", "zoo.val_tricky"] + VAL_PARAM

# typical student networks (the notebook's baseline DQN with the changes students make): other teams in the round robin
TRAIN_STUDENTS = sorted(f"net:runs/students/{p.name}" for p in (ROOT / "runs" / "students").glob("train_*") if (p / "submission.pth").exists())
TEST_STUDENTS = sorted(f"net:runs/students/{p.name}" for p in (ROOT / "runs" / "students").glob("test_*") if (p / "submission.pth").exists())

TEST: list[str] = ["zoo.test_smart", "zoo.test_boss", "zoo.test_camper"] + TEST_STUDENTS


def exists(spec: str) -> bool:
    if spec.startswith("noisy:"):
        return exists(spec.split(":", 2)[2])
    if spec.startswith("zoo."):
        return (ROOT / "arena" / "zoo" / f"{spec[4:].partition(':')[0]}.py").exists()
    if spec.startswith("net:"):
        folder = Path(spec[4:].removesuffix("@sample"))
        folder = folder if folder.is_absolute() else ROOT / folder
        return (folder / "submission.pth").exists()
    return spec in ("random", "simple") or Path(spec).exists()


def available(specs: list[str]) -> list[str]:
    return [s for s in specs if exists(s)]


def test_panel() -> list[str]:
    return TEST or sorted(f"zoo.{p.stem}" for p in (ROOT / "arena" / "zoo").glob("test_*.py"))


PANELS = {"baselines": lambda: BASELINES, "train": lambda: TRAIN, "val": lambda: VAL, "test": test_panel,
          "experts": lambda: EXPERTS, "students": lambda: TRAIN_STUDENTS, "all": lambda: TRAIN + VAL + test_panel()}


def panel(name: str) -> list[str]:
    """'val', 'train', 'test', 'all', or several joined by '+': the specs that exist."""
    out: list[str] = []
    for part in name.split("+"):
        for s in available(PANELS[part]()):
            if s not in out:
                out.append(s)
    return out
