#!/bin/bash
# The 12 simulated student networks (6 for training, 6 held out), trained in parallel on one GPU.
#   bash arena/students/launch_all.sh [device] [max_minutes]
# Each run: 1 CPU process (1 thread). Logs: runs/students/<name>/train.log
cd /home/minjun_dev/bomb || exit 1
DEV=${1:-cuda:2}
MAXMIN=${2:-25}
run() {  # run <name> <args...>
  local name=$1; shift
  mkdir -p runs/students/$name
  nice -n 10 .venv/bin/python -m arena.students.train_dqn --name "$name" --device "$DEV" --max-minutes "$MAXMIN" "$@" \
    > runs/students/$name/stdout.log 2>&1 &
}
MIX_A="simple:0.5,random:0.25,cautious:0.25"
MIX_B="simple:0.34,random:0.33,cautious:0.33"

# ---- training set (seeds 0-5) -------------------------------------------------------------------------------
run train_1_base_s0          --seed 0                                    # the notebook as it is
run train_2_ep4k_selfpen_s1  --seed 1 --episodes 4000 --eps-decay 2000 --reward selfpen
run train_3_ep10k_ddqn_pot_s2 --seed 2 --episodes 10000 --eps-decay 3000 --double --reward pot
run train_4_res_mix_win_s3   --seed 3 --episodes 4000 --eps-decay 2000 --arch res --opp $MIX_A --reward win
run train_5_selfplay_s4      --seed 4 --episodes 3000 --eps-decay 1500 --opp self:0.5,simple:0.5 --reward selfpen
run train_6_bc_dqn_s5        --seed 5 --bc-games 2000 --bc-epochs 20 --episodes 1500 --eps-start 0.2 --eps-decay 750 \
                             --opp simple:0.4,random:0.2,cautious:0.4 --reward selfpen \
                             --pick-best --eval-every 250 --eval-opp simple,cautious

# ---- held-out test set (seeds 10-15), never used for training -----------------------------------------------
run test_1_base_s10          --seed 10                                   # the notebook as it is, another seed
run test_2_ep4k_naive_s11    --seed 11 --episodes 4000 --reward naive    # more games, epsilon schedule untouched
run test_3_ep10k_ddqn_win_s12 --seed 12 --episodes 10000 --eps-decay 5000 --double --reward win
run test_4_res2_ddqn_mix_s13 --seed 13 --episodes 4000 --eps-decay 2000 --arch res2 --double --opp $MIX_B --reward selfpen
run test_5_selfplay_pot_s14  --seed 14 --episodes 3000 --eps-decay 1500 --opp self:0.7,simple:0.15,random:0.15 --reward pot
run test_6_bc_dqfd_res_s15   --seed 15 --arch res --bc-games 1000 --bc-epochs 15 --bc-noise 0.1 --bc-loss margin \
                             --margin 0.8 --demo-frac 0.25 --episodes 1500 --eps-start 0.1 --eps-decay 750 \
                             --opp simple:0.3,random:0.2,cautious:0.5 --reward win \
                             --pick-best --eval-every 250 --eval-opp simple,cautious
wait
echo "all 12 runs finished: $(date)"
