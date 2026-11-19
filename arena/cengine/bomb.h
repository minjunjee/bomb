/* bomb.h - the BOMB competition game engine and the fixed network preprocess in plain C11.
 *
 * Exactly equivalent to bombcompetition/game.py (Game.step, Game.obs order, the sudden-death shrink) and to
 * bombcompetition/neural.py (preprocess(turn(obs))) for FIXED fuses (the league rules). Differential tests:
 * arena/cengine/test_equiv.py.
 *
 * Quick tour for a search program written in C (compile bomb.c into it, or link libbomb.so):
 *
 *     BombState s, child;
 *     bomb_init(&s, walls81, BOMB_CELL(1, 1), BOMB_CELL(7, 7), NULL);     // NULL = the league rules
 *     memcpy(&child, &s, sizeof s);                                      // a state is one flat POD block
 *     int f = bomb_step(&child, BOMB_UP, BOMB_BOMB);                     // >= 0: BOMB_FLAG_* bits, < 0: BOMB_ERR_*
 *     if (f & BOMB_FLAG_HIT1) ...                                        // player 1 was hit in this step
 *     int legal = bomb_effective_mask(&s, 0);                            // bit a set <=> action a does something
 *     uint8_t when[BOMB_CELLS];
 *     bomb_blast_times(&s, when, NULL);                                  // exact, chains + shrink included
 *
 * Conventions
 *   - The board is always 9x9. A cell is one byte: cell = row * 9 + col (row 0 = top). The border must be wall.
 *   - Actions: 0 STAY, 1 UP (row - 1), 2 DOWN (row + 1), 3 LEFT (col - 1), 4 RIGHT (col + 1), 5 BOMB.
 *   - Players are 0 (A) and 1 (B).
 *   - bombs[0 .. n_bombs) is kept in the order Game.obs() returns them (insertion order of the Python dict).
 *     `fuse` is what obs shows: the number of steps until the bomb explodes (1 = at the end of the step being
 *     decided now), `age` the number of steps since it was dropped.
 *   - States are canonical: unused bomb slots and padding are zero, so two equal game states are equal with
 *     memcmp (handy for transposition tables). Keep it that way if you build states by hand.
 *   - No function allocates memory or keeps global state (two read-only tables are filled when the library is
 *     loaded), so everything is thread safe as long as each thread works on its own states.
 *   - Compile WITHOUT -ffast-math: bomb_features reproduces the float32 bit patterns of the Python preprocess.
 */
#ifndef BOMB_H
#define BOMB_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BOMB_N          9
#define BOMB_CELLS      81
#define BOMB_CAP        16                     /* capacity of the bomb list: rules.max_bombs <= BOMB_CAP / 2 */
#define BOMB_CHANNELS   13
#define BOMB_FEAT_SIZE  (BOMB_CHANNELS * BOMB_CELLS)

#define BOMB_CELL(r, c) ((r) * BOMB_N + (c))
#define BOMB_ROW(cell)  ((cell) / BOMB_N)
#define BOMB_COL(cell)  ((cell) % BOMB_N)

enum { BOMB_STAY = 0, BOMB_UP = 1, BOMB_DOWN = 2, BOMB_LEFT = 3, BOMB_RIGHT = 4, BOMB_BOMB = 5, BOMB_ACTIONS = 6 };

/* grid[cell] bits. grid[cell] == 0 <=> a cell one can walk onto (if the other player is not standing there). */
#define BOMB_F_WALL 1
#define BOMB_F_BOMB 2

/* bits of the value returned by bomb_step / bomb_step_info */
#define BOMB_FLAG_HIT0        0x0001  /* player 0 was inside an explosion (lost a life) */
#define BOMB_FLAG_HIT1        0x0002  /* player 1 was inside an explosion */
#define BOMB_FLAG_KILLED0     0x0004  /* player 0 was hit and a bomb of player 1 was part of it: Death(0, killer=1) */
#define BOMB_FLAG_KILLED1     0x0008  /* player 1 was hit and a bomb of player 0 was part of it: Death(1, killer=0) */
#define BOMB_FLAG_DONE        0x0010  /* the game is over after this step */
#define BOMB_FLAG_SHRINK      0x0020  /* a ring turned into walls in this step */
#define BOMB_FLAG_TELEPORT0   0x0040  /* player 0 stood on the closing ring and was moved inside */
#define BOMB_FLAG_TELEPORT1   0x0080
#define BOMB_FLAG_CHAIN       0x0100  /* a bomb was set off by another one before its own fuse ran out */
#define BOMB_FLAG_COLLISION   0x0200  /* both players wanted the same cell, nobody moved */
#define BOMB_FLAG_NEW_ROUND   0x0400  /* somebody was hit but the game goes on (lives > 1): back to the start cells */
#define BOMB_FLAG_DROP0       0x0800  /* player 0 dropped a bomb */
#define BOMB_FLAG_DROP1       0x1000
#define BOMB_FLAG_EXPLOSION   0x2000  /* at least one bomb exploded */
#define BOMB_FLAG_SHRINK_BOMB 0x4000  /* the shrink removed at least one bomb */

#define BOMB_ERR_DONE   (-1)   /* step on a finished game (Python: RuntimeError); the state is unchanged */
#define BOMB_ERR_ACTION (-2)   /* an action outside 0..5; the state is unchanged */
#define BOMB_ERR_CAP    (-3)   /* bomb list overflow: impossible when rules.max_bombs <= BOMB_CAP / 2 */
#define BOMB_ERR_ARG    (-4)   /* bomb_init: bad walls / start cells / rules */

typedef struct BombItem {
    uint8_t cell;    /* row * 9 + col */
    uint8_t owner;   /* 0 or 1 */
    int16_t fuse;    /* steps until it explodes (obs["bombs"][i]["fuse"]) */
    int16_t age;     /* steps since it was dropped (obs["bombs"][i]["age"]) */
} BombItem;

typedef struct BombRules {
    int32_t fuse;          /* fuse of a new bomb (league: 5; in the next obs it shows fuse - 1) */
    int32_t radius;        /* blast reach in each direction (2) */
    int32_t max_bombs;     /* bombs per player on the board at once (3) */
    int32_t lives;         /* lives at the start (1) */
    int32_t shrink_every;  /* sudden death every this many steps, 0 = never (100) */
    int32_t max_steps;     /* (300) */
} BombRules;

typedef struct BombState {
    uint8_t   grid[BOMB_CELLS];  /* BOMB_F_WALL | BOMB_F_BOMB per cell */
    uint8_t   n_bombs;
    uint8_t   done;              /* 1 = the game is over */
    uint8_t   pos[2];            /* the players' cells */
    uint8_t   starts[2];         /* start cells (a new round after a hit when lives > 1) */
    uint8_t   pad_;              /* always 0 */
    BombItem  bombs[BOMB_CAP];   /* [0, n_bombs) in obs order; the rest is zero */
    int32_t   t;                 /* the step counter (obs["step"]) */
    int32_t   lives[2];
    int32_t   kills[2];
    BombRules rules;
} BombState;

/* What happened in one step (optional output of bomb_step_info). */
typedef struct BombStepInfo {
    int32_t flags;               /* the same value bomb_step_info returns */
    uint8_t hit_pos[2];          /* where the players were at the end of the step, before a new round (Game.last_pos) */
    uint8_t n_exploded;          /* bombs that exploded in this step */
    uint8_t n_blast;             /* number of cells with blast[cell] == 1 */
    uint8_t blast[BOMB_CELLS];   /* 1 = the cell was inside an explosion in this step (Game.last_blast) */
} BombStepInfo;

extern const BombRules BOMB_LEAGUE_RULES;   /* {5, 2, 3, 1, 100, 300} */
extern const uint8_t   BOMB_RING[BOMB_CELLS]; /* 0 = the border, 1 = the ring inside it, ... 4 = the centre */

/* ---- one state ------------------------------------------------------------------------------------------------- */

/* A new game. walls: 81 bytes, non-zero = wall, the border must be wall. rules == NULL: the league rules.
 * Returns 0, or BOMB_ERR_ARG (bad border, start cells equal / on a wall / out of range, max_bombs > BOMB_CAP / 2,
 * a fuse outside int16). */
int bomb_init(BombState *s, const uint8_t *walls, int start0, int start1, const BombRules *rules);

/* Setup helper for building a mid-game state by hand (after bomb_init; then set pos / t / lives / kills / done
 * directly): appends a bomb to the list. Returns 0, or BOMB_ERR_ARG (cell on the border or already holding a
 * bomb, list full, owner not 0 / 1, fuse or age outside int16). */
int bomb_put_bomb(BombState *s, int cell, int owner, int fuse, int age);

/* A state from raw parts (a copy of a game in progress). v[10] = {pos0, pos1, start0, start1, t, lives0, lives1,
 * kills0, kills1, done}; bombs = n_bombs * {cell, owner, fuse, age} in obs order. Unlike bomb_init, cells of
 * players / starts may be walls (start cells are after a shrink) but never on the border. Returns 0 or
 * BOMB_ERR_ARG (then *s is unspecified). */
int bomb_load(BombState *s, const uint8_t *walls, const int32_t *v, const int32_t *bombs, int n_bombs,
              const BombRules *rules);

/* One step, both players at once: Game.step(a0, a1). Returns the BOMB_FLAG_* bits (>= 0) or a BOMB_ERR_* (< 0). */
int bomb_step(BombState *s, int a0, int a1);
/* The same, and fills *info (which cells exploded, ...). */
int bomb_step_info(BombState *s, int a0, int a1, BombStepInfo *info);

/* Game.winner(): the player with more lives left, -1 when both have the same (a draw if the game is over). */
int bomb_winner(const BombState *s);

/* Bit a is set <=> action a is in the reference legal_actions(obs) of `player`: STAY and BOMB always, a move
 * unless the target is a wall, holds a bomb, or is the cell of the other player. */
int bomb_legal_mask(const BombState *s, int player);
/* The actions with pairwise different effects: like bomb_legal_mask, but BOMB only when a bomb would really be
 * dropped (fewer than max_bombs of the player's bombs on the board and none on its cell). A move that is not in
 * the mask acts like STAY; BOMB not in the mask acts like STAY as well. */
int bomb_effective_mask(const BombState *s, int player);

/* The cells a bomb at `cell` covers with this state's walls and radius (blast_cells): writes up to 13 cells to
 * out (the bomb's cell first) and returns how many. */
int bomb_blast_cells(const BombState *s, int cell, uint8_t *out);

/* Chain-adjusted fuses, shrink ignored (exactly the timing behind the "danger" input plane and what most
 * rule-based bots compute): out[i] = the step at whose end bombs[i] explodes (1 = the step being decided now)
 * when no further bomb is dropped. Returns n_bombs. out needs BOMB_CAP entries. */
int bomb_chain_fuses(const BombState *s, int16_t *out);
/* The same as a map: when[cell] = the earliest chain-adjusted fuse among the bombs whose blast covers the cell,
 * 0 = no bomb reaches it. (danger plane = 1 / when.) Values above 255 are stored as 255. */
void bomb_danger_times(const BombState *s, uint8_t *when);

/* The EXACT explosion-time map: plays the bombs forward (chains, the sudden-death shrink that removes bombs and
 * adds walls, the end of the game at max_steps) assuming nobody drops another bomb and nobody is hit.
 *   first[cell] = k  <=> the cell is inside an explosion at the end of the k-th step from now (k = 1 is the step
 *                        being decided now) and not earlier; 0 = never. Values above 255 are stored as 255.
 *   mask[cell]  bit (k - 1) set <=> the cell is inside an explosion at the end of the k-th step (k <= 64).
 * Either pointer may be NULL. Returns the last k with an explosion (0: nothing will explode). */
int bomb_blast_times(const BombState *s, uint8_t *first, uint64_t *mask);

/* ---- the network input ----------------------------------------------------------------------------------------- */

/* out[13 * 81] = preprocess(turn(obs(player))) of bombcompetition/neural.py, bit for bit: channel-major, cell =
 * row * 9 + col in the TURNED frame (for player 1 the board is turned by 180 degrees). Like the reference call
 * preprocess(turn(obs)) this uses the LEAGUE constants (radius 2, max_bombs 3, shrink_every 100, max_steps 300)
 * whatever the rules of the state are. */
void bomb_features(const BombState *s, int player, float *out);
/* preprocess(turn(obs(player)), rules): uses radius, max_bombs (must not be 0), shrink_every (>= 0), max_steps
 * of *rules. */
void bomb_features_rules(const BombState *s, int player, const BombRules *rules, float *out);

/* ---- many states in one contiguous buffer (OpenMP when n_threads > 1; n_threads <= 1 never enters OpenMP) ------- */

/* Reset states[idx[j]] (idx == NULL: states[j]) for j < k from walls[j * 81 ..] and starts[j * 2 ..]
 * (starts == NULL: (1, 1) and (7, 7)). rules == NULL: the league rules. Returns 0, or j + 1 of the first bad map
 * (the states before it are reset, that one and the later ones are untouched). */
int bomb_reset_batch(BombState *states, const int32_t *idx, int k, const uint8_t *walls, const uint8_t *starts,
                     const BombRules *rules);

/* Step every state i with actions[2 * i], actions[2 * i + 1]. A state that is already done (or with
 * active[i] == 0 when active != NULL) is left untouched and reports hit 0 0. Outputs (each may be NULL):
 * hit[2 * i + p] = 1 if player p was hit, done[i], winner[i] (bomb_winner after the step), flags[i] (the
 * bomb_step value; 0 for a skipped state). Returns the number of states that got an invalid action (those are
 * left untouched with flags[i] = BOMB_ERR_ACTION). No auto-reset. */
int bomb_step_batch(BombState *states, int n, const uint8_t *actions, uint8_t *hit, uint8_t *done, int8_t *winner,
                    int32_t *flags, const uint8_t *active, int n_threads);

/* out[i * 13 * 81 ..] = bomb_features(&states[i], players[i]). */
void bomb_features_batch(const BombState *states, int n, const uint8_t *players, float *out, int n_threads);
void bomb_features_rules_batch(const BombState *states, int n, const uint8_t *players, const BombRules *rules,
                               float *out, int n_threads);

/* mask[(2 * i + p) * 6 + a] = 1 if action a is legal for player p in state i (bomb_legal_mask, or
 * bomb_effective_mask when effective != 0). */
void bomb_legal_batch(const BombState *states, int n, uint8_t *mask, int effective, int n_threads);

/* ---- misc -------------------------------------------------------------------------------------------------------- */

/* The float32 the Python preprocess stores for 1 / k (a double division, then rounded to float32). */
float bomb_inv(int k);
/* Sizes and field offsets, so that a foreign-language mirror of the structs can check itself. Writes up to
 * `cap` ints: sizeof(BombState), sizeof(BombItem), sizeof(BombRules), sizeof(BombStepInfo), then the offsets of
 * BombState.{grid, n_bombs, done, pos, starts, bombs, t, lives, kills, rules} and of
 * BombStepInfo.{flags, hit_pos, n_exploded, n_blast, blast}. Returns how many it would write (19). */
int bomb_layout(int32_t *out, int cap);
/* The build id that build.py compiled in ("dev" when built by hand). */
const char *bomb_build_id(void);
/* Benchmark helper: plays `steps` engine steps from copies of *init (a finished game restarts from *init) with
 * random actions (BOMB with probability bomb_pct / 100, otherwise STAY or a move, uniformly) and returns a
 * checksum; *games (may be NULL) receives the number of finished games. with_features != 0 also computes both
 * players' features after every step. */
uint64_t bomb_bench(const BombState *init, int64_t steps, uint64_t seed, int bomb_pct, int with_features,
                    int64_t *games);

#ifdef __cplusplus
}
#endif
#endif /* BOMB_H */
