/* bomb.c - see bomb.h. Plain C11, no dependencies; OpenMP is used only inside the *_batch functions.
 *
 * Every rule below is written next to the line of bombcompetition/game.py or neural.py it mirrors.
 * Do not compile with -ffast-math (bomb_features must give the reference's float32 bit patterns).
 */
#include "bomb.h"

#include <stddef.h>
#include <stdlib.h>

#ifndef BOMB_BUILD_ID
#define BOMB_BUILD_ID "dev"
#endif
/* build.py looks for this marker in the .so to know which sources / flags / host it was built from */
static const char BUILD_ID_STR[] __attribute__((used)) = "BOMB_BUILD_ID=" BOMB_BUILD_ID ";";

const BombRules BOMB_LEAGUE_RULES = {5, 2, 3, 1, 100, 300};

/* ring_of(cell, 9) = min(row, col, 8 - row, 8 - col) */
const uint8_t BOMB_RING[BOMB_CELLS] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 2, 2, 2, 2, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 2, 2, 2, 2, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static const int DELTA[BOMB_ACTIONS] = {0, -BOMB_N, BOMB_N, -1, 1, 0}; /* DELTAS: STAY UP DOWN LEFT RIGHT BOMB */
static const int DIRS[4] = {-BOMB_N, BOMB_N, -1, 1};

/* INV[k] = the float32 that numpy stores when the Python float 1 / k is assigned into a float32 array:
 * a double division, then one rounding to float32. */
#define INV_N 512
static float INV[INV_N];

static inline float inv_slow(int k)
{
    volatile double d = 1.0 / (double)k; /* volatile: really a double division followed by a narrowing */
    return (float)d;
}

__attribute__((constructor)) static void bomb_fill_tables(void)
{
    INV[0] = 0.0f;
    for (int k = 1; k < INV_N; k++) INV[k] = inv_slow(k);
}

static inline float inv_f32(int k) { return (k > 0 && k < INV_N) ? INV[k] : inv_slow(k); }

float bomb_inv(int k) { return inv_f32(k); }

const char *bomb_build_id(void) { return BOMB_BUILD_ID; }

int bomb_layout(int32_t *out, int cap)
{
    const int32_t v[] = {
        (int32_t)sizeof(BombState), (int32_t)sizeof(BombItem), (int32_t)sizeof(BombRules),
        (int32_t)sizeof(BombStepInfo),
        (int32_t)offsetof(BombState, grid), (int32_t)offsetof(BombState, n_bombs), (int32_t)offsetof(BombState, done),
        (int32_t)offsetof(BombState, pos), (int32_t)offsetof(BombState, starts), (int32_t)offsetof(BombState, bombs),
        (int32_t)offsetof(BombState, t), (int32_t)offsetof(BombState, lives), (int32_t)offsetof(BombState, kills),
        (int32_t)offsetof(BombState, rules),
        (int32_t)offsetof(BombStepInfo, flags), (int32_t)offsetof(BombStepInfo, hit_pos),
        (int32_t)offsetof(BombStepInfo, n_exploded), (int32_t)offsetof(BombStepInfo, n_blast),
        (int32_t)offsetof(BombStepInfo, blast),
    };
    const int n = (int)(sizeof v / sizeof v[0]);
    for (int i = 0; i < n && i < cap; i++) out[i] = v[i];
    return n;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* small helpers                                                                                                    */
/* ---------------------------------------------------------------------------------------------------------------- */

/* blast_cells(walls, cell, radius): the cell itself, then up to `radius` cells in each direction, stopped by
 * walls only. At most 13 cells (the border is wall, so a ray never leaves the board). */
static inline int blast_list(const uint8_t *grid, int cell, int radius, uint8_t *out)
{
    int n = 0;
    out[n++] = (uint8_t)cell;
    for (int d = 0; d < 4; d++) {
        const int st = DIRS[d];
        int x = cell;
        for (int k = 0; k < radius; k++) {
            x += st;
            if (grid[x] & BOMB_F_WALL) break;
            out[n++] = (uint8_t)x;
        }
    }
    return n;
}

static inline int bomb_index(const BombState *s, int cell)
{
    for (int j = 0; j < s->n_bombs; j++)
        if (s->bombs[j].cell == cell) return j;
    return 0; /* unreachable while grid and bombs[] agree */
}

/* Remove the bombs whose bit is set, keeping the order of the others (dict.pop keeps the insertion order). */
static inline void remove_bombs(BombState *s, uint32_t gone)
{
    const int n = s->n_bombs;
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (gone >> i & 1u) {
            s->grid[s->bombs[i].cell] &= (uint8_t)~BOMB_F_BOMB;
        } else {
            if (m != i) s->bombs[m] = s->bombs[i];
            m++;
        }
    }
    for (int i = m; i < n; i++) memset(&s->bombs[i], 0, sizeof(BombItem)); /* keep the state canonical */
    s->n_bombs = (uint8_t)m;
}

static inline int clamp_radius(int r) { return r < 0 ? 0 : (r > BOMB_N ? BOMB_N : r); }

/* Game._explode: every fuse -1 / age +1, then the bombs at 0 explode and set off the bombs they reach.
 * p0, p1: the players' cells (pass -1 to ignore). hit_by[p]: bit o set if a bomb of owner o covered player p.
 * blast (may be NULL): set to 1 on exploded cells. Returns the bit mask (old indices) of the exploded bombs. */
static inline uint32_t tick_explode(BombState *s, int p0, int p1, int *hit_by, uint8_t *blast, int *flags)
{
    const int n = s->n_bombs;
    uint32_t expl = 0;
    int stack[BOMB_CAP], sp = 0;
    for (int i = 0; i < n; i++) {
        BombItem *b = &s->bombs[i];
        b->fuse--;
        b->age++;
        if (b->fuse <= 0) {
            expl |= 1u << i;
            stack[sp++] = i;
        }
    }
    if (!sp) return 0;
    const uint8_t *grid = s->grid;
    const int radius = s->rules.radius;
    while (sp) {
        const int i = stack[--sp];
        const int c = s->bombs[i].cell, ob = 1 << s->bombs[i].owner;
        if (blast) blast[c] = 1;
        if (c == p0) hit_by[0] |= ob;
        if (c == p1) hit_by[1] |= ob;
        for (int d = 0; d < 4; d++) {
            const int st = DIRS[d];
            int x = c;
            for (int k = 0; k < radius; k++) {
                x += st;
                const int g = grid[x];
                if (g & BOMB_F_WALL) break; /* walls stop a blast, bombs and players do not */
                if (blast) blast[x] = 1;
                if (x == p0) hit_by[0] |= ob;
                if (x == p1) hit_by[1] |= ob;
                if (g & BOMB_F_BOMB) {
                    const int j = bomb_index(s, x);
                    if (!(expl >> j & 1u)) { /* a bomb inside an explosion explodes in the same step */
                        expl |= 1u << j;
                        stack[sp++] = j;
                        *flags |= BOMB_FLAG_CHAIN;
                    }
                }
            }
        }
    }
    remove_bombs(s, expl);
    *flags |= BOMB_FLAG_EXPLOSION;
    return expl;
}

/* Game._shrink(k): ring k becomes wall, bombs on it vanish, players on it move to the nearest open cell inside. */
static inline void shrink(BombState *s, int k, int with_players, int *flags)
{
    if (k > 4) return; /* shrink_ring: no ring k on a 9x9 board and nothing inside it */
    uint8_t *grid = s->grid;
    uint8_t inner[BOMB_CELLS];
    int n_inner = 0;
    for (int c = 0; c < BOMB_CELLS; c++) /* row-major order = sorted by (row, col) */
        if (BOMB_RING[c] > k && !(grid[c] & BOMB_F_WALL)) inner[n_inner++] = (uint8_t)c;
    if (!n_inner) return; /* nothing inside is open: the map stops shrinking */
    uint32_t gone = 0;
    for (int c = 0; c < BOMB_CELLS; c++) {
        if (BOMB_RING[c] != k) continue;
        if (grid[c] & BOMB_F_BOMB) gone |= 1u << bomb_index(s, c);
        grid[c] |= BOMB_F_WALL;
    }
    if (gone) {
        remove_bombs(s, gone);
        *flags |= BOMB_FLAG_SHRINK_BOMB;
    }
    *flags |= BOMB_FLAG_SHRINK;
    if (!with_players) return;
    for (int p = 0; p < 2; p++) { /* player 0 first, then player 1 (who must avoid player 0's NEW cell) */
        const int c0 = s->pos[p];
        if (BOMB_RING[c0] != k) continue;
        const int r0 = c0 / BOMB_N, q0 = c0 % BOMB_N, other = s->pos[1 - p];
        int best = -1, best_d = 1 << 30;
        for (int i = 0; i < n_inner; i++) { /* free = inner cells without a bomb and not the other player's */
            const int x = inner[i];
            if ((grid[x] & BOMB_F_BOMB) || x == other) continue;
            const int d = abs(x / BOMB_N - r0) + abs(x % BOMB_N - q0);
            if (d < best_d) { /* strict: ties keep the smaller (row, col) */
                best_d = d;
                best = x;
            }
        }
        if (best < 0) { /* "or inner": nothing free, take any inner open cell */
            for (int i = 0; i < n_inner; i++) {
                const int x = inner[i];
                const int d = abs(x / BOMB_N - r0) + abs(x % BOMB_N - q0);
                if (d < best_d) {
                    best_d = d;
                    best = x;
                }
            }
        }
        s->pos[p] = (uint8_t)best;
        *flags |= BOMB_FLAG_TELEPORT0 << p;
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* one state                                                                                                        */
/* ---------------------------------------------------------------------------------------------------------------- */

static int rules_ok(const BombRules *rules)
{
    return rules->max_bombs <= BOMB_CAP / 2 && rules->fuse <= 32000 && rules->fuse >= -32000;
}

static void set_board(BombState *s, const uint8_t *walls, const BombRules *rules)
{
    memset(s, 0, sizeof *s);
    for (int c = 0; c < BOMB_CELLS; c++) s->grid[c] = walls[c] ? BOMB_F_WALL : 0;
    s->rules = *rules;
    s->rules.radius = clamp_radius(rules->radius);            /* a longer ray ends at the border anyway */
    if (s->rules.shrink_every < 0) s->rules.shrink_every = 0; /* never */
}

static int border_ok(const uint8_t *walls)
{
    for (int c = 0; c < BOMB_CELLS; c++)
        if (BOMB_RING[c] == 0 && !walls[c]) return 0; /* the map must have walls all around */
    return 1;
}

int bomb_init(BombState *s, const uint8_t *walls, int start0, int start1, const BombRules *rules)
{
    if (!rules) rules = &BOMB_LEAGUE_RULES;
    if (!rules_ok(rules) || !border_ok(walls)) return BOMB_ERR_ARG;
    if (start0 < 0 || start0 >= BOMB_CELLS || start1 < 0 || start1 >= BOMB_CELLS || start0 == start1 ||
        walls[start0] || walls[start1])
        return BOMB_ERR_ARG; /* bad start cells */
    set_board(s, walls, rules);
    s->pos[0] = s->starts[0] = (uint8_t)start0;
    s->pos[1] = s->starts[1] = (uint8_t)start1;
    s->lives[0] = s->lives[1] = rules->lives;
    return 0;
}

int bomb_put_bomb(BombState *s, int cell, int owner, int fuse, int age)
{
    if (cell < 0 || cell >= BOMB_CELLS || BOMB_RING[cell] == 0 || (s->grid[cell] & BOMB_F_BOMB) ||
        s->n_bombs >= BOMB_CAP || (owner != 0 && owner != 1) || fuse > 32000 || fuse < -32000 || age > 32000 ||
        age < -32000)
        return BOMB_ERR_ARG;
    BombItem *b = &s->bombs[s->n_bombs++];
    b->cell = (uint8_t)cell;
    b->owner = (uint8_t)owner;
    b->fuse = (int16_t)fuse;
    b->age = (int16_t)age;
    s->grid[cell] |= BOMB_F_BOMB;
    return 0;
}

int bomb_load(BombState *s, const uint8_t *walls, const int32_t *v, const int32_t *bombs, int n_bombs,
              const BombRules *rules)
{
    if (!rules) rules = &BOMB_LEAGUE_RULES;
    if (!rules_ok(rules) || !border_ok(walls) || n_bombs < 0 || n_bombs > BOMB_CAP) return BOMB_ERR_ARG;
    for (int i = 0; i < 4; i++) /* players and start cells must be off the border (rays and moves rely on it) */
        if (v[i] < 0 || v[i] >= BOMB_CELLS || BOMB_RING[v[i]] == 0) return BOMB_ERR_ARG;
    set_board(s, walls, rules);
    s->pos[0] = (uint8_t)v[0];
    s->pos[1] = (uint8_t)v[1];
    s->starts[0] = (uint8_t)v[2];
    s->starts[1] = (uint8_t)v[3];
    s->t = v[4];
    s->lives[0] = v[5];
    s->lives[1] = v[6];
    s->kills[0] = v[7];
    s->kills[1] = v[8];
    s->done = v[9] != 0;
    int count[2] = {0, 0};
    for (int i = 0; i < n_bombs; i++) {
        const int32_t *b = bombs + 4 * i;
        if (bomb_put_bomb(s, b[0], b[1], b[2], b[3]) != 0) return BOMB_ERR_ARG;
        count[b[1]]++;
    }
    if (count[0] > BOMB_CAP / 2 || count[1] > BOMB_CAP / 2) return BOMB_ERR_ARG; /* later drops could overflow */
    return 0;
}

static inline int step_impl(BombState *s, int a0, int a1, BombStepInfo *info)
{
    if (s->done) return BOMB_ERR_DONE;
    if ((unsigned)a0 >= BOMB_ACTIONS || (unsigned)a1 >= BOMB_ACTIONS) return BOMB_ERR_ACTION;
    int flags = 0;
    uint8_t *grid = s->grid;

    /* 1. Drop: player 0 then player 1; not on a cell with a bomb; at most max_bombs per owner. */
    if (a0 == BOMB_BOMB || a1 == BOMB_BOMB) {
        for (int p = 0; p < 2; p++) {
            if ((p ? a1 : a0) != BOMB_BOMB) continue;
            const int c = s->pos[p];
            if (grid[c] & BOMB_F_BOMB) continue;
            int mine = 0;
            for (int i = 0; i < s->n_bombs; i++) mine += (s->bombs[i].owner == p);
            if (mine >= s->rules.max_bombs) continue;
            if (s->n_bombs >= BOMB_CAP) return BOMB_ERR_CAP;
            BombItem *b = &s->bombs[s->n_bombs++];
            b->cell = (uint8_t)c;
            b->owner = (uint8_t)p;
            b->fuse = (int16_t)s->rules.fuse;
            b->age = 0;
            grid[c] |= BOMB_F_BOMB;
            flags |= BOMB_FLAG_DROP0 << p;
        }
    }

    /* 2. Move: blocked by a wall, a bomb (also one dropped just now) or the cell the other player stands on at
     *    the START of the step; equal targets -> nobody moves. */
    const int p0 = s->pos[0], p1 = s->pos[1];
    int t0 = p0 + DELTA[a0], t1 = p1 + DELTA[a1];
    if (t0 != p0 && (grid[t0] != 0 || t0 == p1)) t0 = p0;
    if (t1 != p1 && (grid[t1] != 0 || t1 == p0)) t1 = p1;
    if (t0 != t1) {
        s->pos[0] = (uint8_t)t0;
        s->pos[1] = (uint8_t)t1;
    } else {
        if (t0 != p0) flags |= BOMB_FLAG_COLLISION;
        t0 = p0;
        t1 = p1;
    }

    /* 3. Tick and explode. */
    int hit_by[2] = {0, 0};
    uint32_t expl = 0;
    if (info) memset(info->blast, 0, BOMB_CELLS);
    if (s->n_bombs) expl = tick_explode(s, t0, t1, hit_by, info ? info->blast : NULL, &flags);

    /* 4. Hits: Death(p, killer = the opponent if one of its bombs covered p, else None). */
    int dead = 0;
    for (int p = 0; p < 2; p++) {
        if (!hit_by[p]) continue;
        dead = 1;
        s->lives[p]--;
        flags |= BOMB_FLAG_HIT0 << p;
        if (hit_by[p] >> (1 - p) & 1) {
            s->kills[1 - p]++;
            flags |= BOMB_FLAG_KILLED0 << p;
        }
    }

    s->t++;
    const int done = (s->lives[0] <= 0) | (s->lives[1] <= 0) | (s->t >= s->rules.max_steps);
    s->done = (uint8_t)done;

    /* 5. Sudden death. */
    const int every = s->rules.shrink_every;
    if (!done && every > 0 && s->t % every == 0) shrink(s, s->t / every, 1, &flags);

    if (info) {
        info->hit_pos[0] = s->pos[0];
        info->hit_pos[1] = s->pos[1];
        info->n_exploded = (uint8_t)__builtin_popcount(expl);
        int nb = 0;
        for (int c = 0; c < BOMB_CELLS; c++) nb += info->blast[c];
        info->n_blast = (uint8_t)nb;
    }

    /* A hit without the end of the game (lives > 1): a new round, start cells and no bombs. */
    if (dead && !done) {
        for (int i = 0; i < s->n_bombs; i++) grid[s->bombs[i].cell] &= (uint8_t)~BOMB_F_BOMB;
        memset(s->bombs, 0, sizeof s->bombs);
        s->n_bombs = 0;
        s->pos[0] = s->starts[0];
        s->pos[1] = s->starts[1];
        flags |= BOMB_FLAG_NEW_ROUND;
    }
    if (done) flags |= BOMB_FLAG_DONE;
    if (info) info->flags = flags;
    return flags;
}

int bomb_step(BombState *s, int a0, int a1) { return step_impl(s, a0, a1, NULL); }

int bomb_step_info(BombState *s, int a0, int a1, BombStepInfo *info) { return step_impl(s, a0, a1, info); }

static inline int winner_of(const BombState *s)
{
    if (s->lives[0] == s->lives[1]) return -1;
    return s->lives[1] > s->lives[0];
}

int bomb_winner(const BombState *s) { return winner_of(s); }

static inline int legal_mask(const BombState *s, int player, int effective)
{
    const int me = s->pos[player], opp = s->pos[1 - player];
    int m = 1 << BOMB_STAY;
    for (int a = BOMB_UP; a <= BOMB_RIGHT; a++) {
        const int t = me + DELTA[a];
        if (s->grid[t] == 0 && t != opp) m |= 1 << a;
    }
    if (!effective) return m | (1 << BOMB_BOMB);
    if (!(s->grid[me] & BOMB_F_BOMB)) {
        int mine = 0;
        for (int i = 0; i < s->n_bombs; i++) mine += (s->bombs[i].owner == player);
        if (mine < s->rules.max_bombs) m |= 1 << BOMB_BOMB;
    }
    return m;
}

int bomb_legal_mask(const BombState *s, int player) { return legal_mask(s, player != 0, 0); }

int bomb_effective_mask(const BombState *s, int player) { return legal_mask(s, player != 0, 1); }

int bomb_blast_cells(const BombState *s, int cell, uint8_t *out)
{
    return blast_list(s->grid, cell, s->rules.radius, out);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* explosion times                                                                                                  */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct {
    int tf[BOMB_CAP];         /* chain-adjusted fuse of every bomb */
    int nb[BOMB_CAP];         /* number of cells in its blast */
    uint8_t bl[BOMB_CAP][16]; /* the cells of its blast */
} Chain;

/* neural.preprocess:
 *     t = {b["pos"]: b["fuse"] for b in obs["bombs"]}
 *     for _ in range(len(t)):
 *         for c in t:
 *             for d in blast_cells(walls, c, radius):
 *                 if d in t: t[d] = min(t[d], t[c])
 * = the least fixed point of "a bomb inside the blast of bomb c explodes no later than c"; the relaxation below
 * runs until nothing changes and reaches the same fixed point. */
static inline void chain_compute(const BombState *s, int radius, Chain *ch)
{
    const uint8_t *grid = s->grid;
    const int n = s->n_bombs;
    uint32_t adj[BOMB_CAP];
    uint32_t any = 0;
    for (int i = 0; i < n; i++) {
        ch->tf[i] = s->bombs[i].fuse;
        const int m = blast_list(grid, s->bombs[i].cell, radius, ch->bl[i]);
        ch->nb[i] = m;
        uint32_t a = 0;
        for (int q = 1; q < m; q++)
            if (grid[ch->bl[i][q]] & BOMB_F_BOMB) a |= 1u << bomb_index(s, ch->bl[i][q]);
        adj[i] = a;
        any |= a;
    }
    if (!any) return;
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = 0; i < n; i++) {
            const int ti = ch->tf[i];
            for (uint32_t a = adj[i]; a; a &= a - 1) {
                const int j = __builtin_ctz(a);
                if (ch->tf[j] > ti) {
                    ch->tf[j] = ti;
                    changed = 1;
                }
            }
        }
    }
}

int bomb_chain_fuses(const BombState *s, int16_t *out)
{
    Chain ch;
    chain_compute(s, s->rules.radius, &ch);
    for (int i = 0; i < s->n_bombs; i++) out[i] = (int16_t)ch.tf[i];
    return s->n_bombs;
}

void bomb_danger_times(const BombState *s, uint8_t *when)
{
    Chain ch;
    chain_compute(s, s->rules.radius, &ch);
    memset(when, 0, BOMB_CELLS);
    for (int i = 0; i < s->n_bombs; i++) {
        const int t = ch.tf[i] > 255 ? 255 : (ch.tf[i] < 1 ? 1 : ch.tf[i]);
        for (int q = 0; q < ch.nb[i]; q++) {
            const int c = ch.bl[i][q];
            if (!when[c] || t < when[c]) when[c] = (uint8_t)t;
        }
    }
}

int bomb_blast_times(const BombState *s, uint8_t *first, uint64_t *mask)
{
    BombState w;
    memcpy(&w, s, sizeof w);
    if (first) memset(first, 0, BOMB_CELLS);
    if (mask) memset(mask, 0, BOMB_CELLS * sizeof(uint64_t));
    const int every = w.rules.shrink_every;
    int last = 0, flags = 0, hit_by[2] = {0, 0};
    uint8_t blast[BOMB_CELLS];
    for (int k = 1; w.n_bombs > 0 && w.t < w.rules.max_steps; k++) {
        memset(blast, 0, BOMB_CELLS);
        if (tick_explode(&w, -1, -1, hit_by, blast, &flags)) {
            last = k;
            for (int c = 0; c < BOMB_CELLS; c++) {
                if (!blast[c]) continue;
                if (first && !first[c]) first[c] = (uint8_t)(k > 255 ? 255 : k);
                if (mask && k <= 64) mask[c] |= 1ull << (k - 1);
            }
        }
        w.t++;
        if (w.t < w.rules.max_steps && every > 0 && w.t % every == 0) shrink(&w, w.t / every, 0, &flags);
    }
    return last;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the network input: neural.preprocess(neural.turn(obs))                                                           */
/* ---------------------------------------------------------------------------------------------------------------- */

enum { CH_WALLS, CH_ME, CH_OPP, CH_MY_BOMBS, CH_OPP_BOMBS, CH_DANGER, CH_MY_REACH, CH_OPP_REACH, CH_MY_LEFT,
       CH_FREE, CH_MY_DIST, CH_OPP_DIST, CH_CLOSING };

/* turn(): player 1 sees the board turned by 180 degrees, cell -> 80 - cell. Every plane is a per-cell fact that
 * turns with the board, so the planes are computed in the real frame and written through this map. */
#define OUT_CELL(c) (flip ? (BOMB_CELLS - 1 - (c)) : (c))

/* _steps(walls, start, bombs | {other}): BFS over cells that are neither wall nor bomb nor the other player's
 * cell (the start itself is always allowed); plane[cell] = 1 / (1 + steps). */
static inline void dist_plane(const uint8_t *grid, int start, int other, int flip, float *plane)
{
    int8_t dist[BOMB_CELLS];
    uint8_t queue[BOMB_CELLS];
    int head = 0, tail = 0;
    memset(dist, -1, sizeof dist);
    dist[start] = 0;
    queue[tail++] = (uint8_t)start;
    plane[OUT_CELL(start)] = INV[1];
    while (head < tail) {
        const int c = queue[head++];
        const int d = dist[c] + 1;
        for (int k = 0; k < 4; k++) {
            const int x = c + DIRS[k];
            if (grid[x] == 0 && x != other && dist[x] < 0) {
                dist[x] = (int8_t)d;
                queue[tail++] = (uint8_t)x;
                plane[OUT_CELL(x)] = INV[1 + d];
            }
        }
    }
}

static inline void features_impl(const BombState *s, int player, int radius, int max_bombs, int every,
                                 int max_steps, float *out)
{
    const uint8_t *grid = s->grid;
    const int flip = player != 0;
    const int me = s->pos[flip], opp = s->pos[1 - flip];
    const int n = s->n_bombs;
    radius = clamp_radius(radius);
    memset(out, 0, sizeof(float) * BOMB_FEAT_SIZE);

    /* walls: 1 = wall.  free: 1 = no wall and no bomb. */
    float *walls = out + CH_WALLS * BOMB_CELLS, *free_ = out + CH_FREE * BOMB_CELLS;
    for (int c = 0; c < BOMB_CELLS; c++) {
        const int g = grid[c], o = OUT_CELL(c);
        walls[o] = (float)(g & BOMB_F_WALL);
        free_[o] = g == 0 ? 1.0f : 0.0f;
    }
    out[CH_ME * BOMB_CELLS + OUT_CELL(me)] = 1.0f;
    out[CH_OPP * BOMB_CELLS + OUT_CELL(opp)] = 1.0f;

    /* danger: max over the bombs that cover the cell of 1 / (chain-adjusted fuse) */
    int mine = 0;
    if (n) {
        Chain ch;
        chain_compute(s, radius, &ch);
        float *danger = out + CH_DANGER * BOMB_CELLS;
        for (int i = 0; i < n; i++) {
            const float v = inv_f32(ch.tf[i]);
            for (int q = 0; q < ch.nb[i]; q++) {
                const int o = OUT_CELL(ch.bl[i][q]);
                if (v > danger[o]) danger[o] = v;
            }
            const int own = s->bombs[i].owner == flip;
            out[(own ? CH_MY_BOMBS : CH_OPP_BOMBS) * BOMB_CELLS + OUT_CELL(s->bombs[i].cell)] = 1.0f;
            mine += own;
        }
    }

    /* my_reach / opp_reach: the blast of a bomb dropped now on my / the opponent's cell */
    uint8_t cells[16];
    int m = blast_list(grid, me, radius, cells);
    for (int q = 0; q < m; q++) out[CH_MY_REACH * BOMB_CELLS + OUT_CELL(cells[q])] = 1.0f;
    m = blast_list(grid, opp, radius, cells);
    for (int q = 0; q < m; q++) out[CH_OPP_REACH * BOMB_CELLS + OUT_CELL(cells[q])] = 1.0f;

    /* my_bombs_left: (max_bombs - mine) / max_bombs, a Python float stored into float32 */
    {
        volatile double q = (double)(max_bombs - mine) / (double)max_bombs;
        const float v = (float)q;
        float *plane = out + CH_MY_LEFT * BOMB_CELLS;
        for (int c = 0; c < BOMB_CELLS; c++) plane[c] = v;
    }

    dist_plane(grid, me, opp, flip, out + CH_MY_DIST * BOMB_CELLS);
    dist_plane(grid, opp, me, flip, out + CH_OPP_DIST * BOMB_CELLS);

    /* closing: the open cells of the next ring to close: 1 / (steps until then) */
    if (every > 0) {
        const int t = s->t;
        const int k = t / every + 1, left = every - t % every;
        if (k <= 4 && (int64_t)t + left < max_steps) {
            int inner = 0;
            for (int c = 0; c < BOMB_CELLS && !inner; c++) inner = BOMB_RING[c] > k && !(grid[c] & BOMB_F_WALL);
            if (inner) {
                const float v = inv_f32(left);
                float *plane = out + CH_CLOSING * BOMB_CELLS;
                for (int c = 0; c < BOMB_CELLS; c++)
                    if (BOMB_RING[c] == k && !(grid[c] & BOMB_F_WALL)) plane[OUT_CELL(c)] = v;
            }
        }
    }
}

void bomb_features(const BombState *s, int player, float *out)
{
    features_impl(s, player, BOMB_LEAGUE_RULES.radius, BOMB_LEAGUE_RULES.max_bombs, BOMB_LEAGUE_RULES.shrink_every,
                  BOMB_LEAGUE_RULES.max_steps, out);
}

void bomb_features_rules(const BombState *s, int player, const BombRules *rules, float *out)
{
    if (!rules) rules = &BOMB_LEAGUE_RULES;
    features_impl(s, player, rules->radius, rules->max_bombs, rules->shrink_every, rules->max_steps, out);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* many states                                                                                                      */
/* ---------------------------------------------------------------------------------------------------------------- */

int bomb_reset_batch(BombState *states, const int32_t *idx, int k, const uint8_t *walls, const uint8_t *starts,
                     const BombRules *rules)
{
    for (int j = 0; j < k; j++) {
        const int s0 = starts ? starts[2 * j] : BOMB_CELL(1, 1);
        const int s1 = starts ? starts[2 * j + 1] : BOMB_CELL(BOMB_N - 2, BOMB_N - 2);
        if (bomb_init(&states[idx ? idx[j] : j], walls + (size_t)j * BOMB_CELLS, s0, s1, rules) != 0) return j + 1;
    }
    return 0;
}

static inline int step_one(BombState *s, const uint8_t *a, uint8_t *hit, uint8_t *done, int8_t *winner,
                           int32_t *flags, int active)
{
    int f = 0, bad = 0;
    if (active && !s->done) {
        f = step_impl(s, a[0], a[1], NULL);
        bad = f == BOMB_ERR_ACTION;
    }
    if (hit) {
        hit[0] = f > 0 && (f & BOMB_FLAG_HIT0);
        hit[1] = f > 0 && (f & BOMB_FLAG_HIT1);
    }
    if (done) *done = s->done;
    if (winner) *winner = (int8_t)winner_of(s);
    if (flags) *flags = f;
    return bad;
}

int bomb_step_batch(BombState *states, int n, const uint8_t *actions, uint8_t *hit, uint8_t *done, int8_t *winner,
                    int32_t *flags, const uint8_t *active, int n_threads)
{
    int bad = 0;
#define STEP_I(i)                                                                                                  \
    step_one(&states[i], actions + 2 * (size_t)(i), hit ? hit + 2 * (size_t)(i) : NULL, done ? done + (i) : NULL,  \
             winner ? winner + (i) : NULL, flags ? flags + (i) : NULL, active ? active[i] : 1)
    if (n_threads <= 1) {
        for (int i = 0; i < n; i++) bad += STEP_I(i);
    } else {
#pragma omp parallel for num_threads(n_threads) schedule(static) reduction(+ : bad)
        for (int i = 0; i < n; i++) bad += STEP_I(i);
    }
#undef STEP_I
    return bad;
}

void bomb_features_batch(const BombState *states, int n, const uint8_t *players, float *out, int n_threads)
{
    bomb_features_rules_batch(states, n, players, &BOMB_LEAGUE_RULES, out, n_threads);
}

void bomb_features_rules_batch(const BombState *states, int n, const uint8_t *players, const BombRules *rules,
                               float *out, int n_threads)
{
    if (!rules) rules = &BOMB_LEAGUE_RULES;
    const int radius = rules->radius, max_bombs = rules->max_bombs, every = rules->shrink_every,
              max_steps = rules->max_steps;
    if (n_threads <= 1) {
        for (int i = 0; i < n; i++)
            features_impl(&states[i], players[i], radius, max_bombs, every, max_steps,
                          out + (size_t)i * BOMB_FEAT_SIZE);
    } else {
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int i = 0; i < n; i++)
            features_impl(&states[i], players[i], radius, max_bombs, every, max_steps,
                          out + (size_t)i * BOMB_FEAT_SIZE);
    }
}

static inline void legal_one(const BombState *s, uint8_t *mask, int effective)
{
    for (int p = 0; p < 2; p++) {
        const int m = legal_mask(s, p, effective);
        for (int a = 0; a < BOMB_ACTIONS; a++) mask[p * BOMB_ACTIONS + a] = (uint8_t)(m >> a & 1);
    }
}

void bomb_legal_batch(const BombState *states, int n, uint8_t *mask, int effective, int n_threads)
{
    if (n_threads <= 1) {
        for (int i = 0; i < n; i++) legal_one(&states[i], mask + (size_t)i * 2 * BOMB_ACTIONS, effective);
    } else {
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int i = 0; i < n; i++) legal_one(&states[i], mask + (size_t)i * 2 * BOMB_ACTIONS, effective);
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* benchmark helper                                                                                                 */
/* ---------------------------------------------------------------------------------------------------------------- */

uint64_t bomb_bench(const BombState *init, int64_t steps, uint64_t seed, int bomb_pct, int with_features,
                    int64_t *games)
{
    BombState s;
    float feat[BOMB_FEAT_SIZE];
    uint64_t x = seed ? seed : 88172645463325252ull, sum = 0;
    int64_t g = 0;
    memcpy(&s, init, sizeof s);
    for (int64_t i = 0; i < steps; i++) {
        int a[2];
        for (int p = 0; p < 2; p++) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            const uint32_t r = (uint32_t)(x >> 24);
            a[p] = (int)(r % 100u) < bomb_pct ? BOMB_BOMB : (int)((r >> 8) % 5u);
        }
        const int f = step_impl(&s, a[0], a[1], NULL);
        sum = sum * 1000003ull + (uint64_t)f + s.pos[0] + ((uint64_t)s.pos[1] << 8) + ((uint64_t)s.n_bombs << 16);
        if (with_features) {
            features_impl(&s, 0, 2, 3, 100, 300, feat);
            sum += (uint64_t)(feat[CH_MY_DIST * BOMB_CELLS + 40] * 1024.0f);
            features_impl(&s, 1, 2, 3, 100, 300, feat);
            sum += (uint64_t)(feat[CH_DANGER * BOMB_CELLS + 40] * 1024.0f);
        }
        if (f & BOMB_FLAG_DONE) {
            memcpy(&s, init, sizeof s);
            g++;
        }
    }
    if (games) *games = g;
    return sum;
}
