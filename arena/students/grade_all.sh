#!/bin/bash
# Check and grade every simulated student network, then one round robin between all of them.
#   bash arena/students/grade_all.sh            (results: runs/students/_results/)
cd /home/minjun_dev/bomb || exit 1
OUT=runs/students/_results
mkdir -p $OUT
NAMES=$(cd runs/students && ls -d train_* test_* 2>/dev/null)
for n in $NAMES; do
  [ -f runs/students/$n/submission.pth ] || { echo "MISSING $n"; continue; }
  .venv/bin/python -c "from arena.nets import check_submission; check_submission('runs/students/$n')" > $OUT/$n.check.txt 2>&1
  nice -n 10 .venv/bin/python -m arena.core leaderboard --me net:runs/students/$n --opp random,simple,zoo.base_cautious \
    --games 100 --procs 16 --json $OUT/$n.leaderboard.json > $OUT/$n.leaderboard.txt 2>&1
  cat $OUT/$n.check.txt $OUT/$n.leaderboard.txt
done
BOTS=$(for n in $NAMES; do printf "net:runs/students/%s," $n; done | sed 's/,$//')
nice -n 10 .venv/bin/python -m arena.core tournament --bots $BOTS --games ${RR_GAMES:-50} --procs 32 \
  --json $OUT/round_robin.json > $OUT/round_robin.txt 2>&1
cat $OUT/round_robin.txt
