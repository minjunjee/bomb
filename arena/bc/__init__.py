"""Behaviour cloning / DAgger: a teacher bot labels states, the network learns the teacher's action.

    arena.bc.gen     parallel data generation with the reference engine      (python -m arena.bc.gen --help)
    arena.bc.train   supervised training on one GPU, export, harness check   (python -m arena.bc.train --help)
    arena.bc.dagger  the resumable DAgger loop built from the two            (python -m arena.bc.dagger --help)
    arena.bc.noise   how often a teacher disagrees with itself; what a student does in those states
    arena.bc.common  the exact uint8 plane codec, the pair mini-language, dataset reading
"""
