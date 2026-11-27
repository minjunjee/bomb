#!/usr/bin/env python3
"""BOJ 18789 (814 - 2) 채점기.

표 파일(숫자 14개짜리 줄 8개)을 읽어, 1부터 X까지 모두 읽히고 X+1은 읽히지 않는 X를 출력한다.
탐색기의 계산과 무관하게, 문제 정의 그대로 한 수씩 확인한다.

사용법: python3 verify.py 표파일
"""
import sys

R, C = 8, 14


def load(path):
    digits = [int(ch) for ch in open(path).read() if ch.isdigit()]
    if len(digits) != R * C:
        sys.exit(f"숫자가 {R * C}개여야 하는데 {len(digits)}개다")
    return digits


def score(grid, limit=10**6):
    # 칸마다 상하좌우와 대각선으로 붙어 있는 칸 목록
    nb = []
    for r in range(R):
        for c in range(C):
            cur = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < R and 0 <= cc < C:
                        cur.append(rr * C + cc)
            nb.append(cur)
    cells_of = [[i for i in range(R * C) if grid[i] == d] for d in range(10)]
    for n in range(1, limit + 1):
        s = [int(ch) for ch in str(n)]
        ends = set(cells_of[s[0]])  # 지금까지 읽은 앞부분이 끝날 수 있는 칸들
        for d in s[1:]:
            ends = {v for u in ends for v in nb[u] if grid[v] == d}
            if not ends:
                break
        if not ends:
            return n - 1
    return limit


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(score(load(sys.argv[1])))
