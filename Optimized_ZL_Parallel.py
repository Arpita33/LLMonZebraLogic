import random
import time
import os
from multiprocessing import Pool
from typing import Optional
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from z3 import *

# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────
Attributes = dict[str, list[str]]
Solution   = dict[int, dict[str, str]]
Clue       = tuple

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_CONFLICTS     = 20 # was 50
#BUCKET_BOUNDARIES = [0, 10, 20, 30, 40, 50]
#BUCKET_BOUNDARIES = [0, 5, 10, 15, 20]
BUCKET_BOUNDARIES = [0, 3, 6, 9, 12, 15]
BUCKET_LABELS     = ["0–3", "3–6", "6–9", "9–12", "12–15"]
BUCKET_COLORS     = ["#4CAF50", "#FFC107", "#FF5722", "#F44336", "#9C27B0"]
CLUE_WEIGHTS      = {
    "FOUNDAT": 5, "NOTAT": 4, "SAMEHOUSE": 3,
    "DIRECTLEFT": 2, "SIDEBYSIDE": 2, "LEFTOF": 1, "RIGHTOF": 1,
}

# ─────────────────────────────────────────────
# 1.  Random solution grid
# ─────────────────────────────────────────────

def random_solution(houses: list[int], attributes: Attributes) -> Solution:
    solution: Solution = {h: {} for h in houses}
    for attr, values in attributes.items():
        shuffled = values[:]
        random.shuffle(shuffled)
        for h, v in zip(houses, shuffled):
            solution[h][attr] = v
    return solution

# ─────────────────────────────────────────────
# 2.  Clue generation
# ─────────────────────────────────────────────

def clue_generation(solution: Solution, houses: list[int], attributes: Attributes) -> list[Clue]:
    clues: list[Clue] = []
    N = len(houses)

    for attr, values in attributes.items():
        for val in values:
            h = next(hh for hh in houses if solution[hh][attr] == val)
            clues.append(("FOUNDAT", attr, val, h))
            for hp in houses:
                if hp != h:
                    clues.append(("NOTAT", attr, val, hp))
            for attr2 in attributes:
                if attr2 == attr:
                    continue
                val2 = solution[h][attr2]
                clues.append(("SAMEHOUSE", attr, val, attr2, val2))
            if h < N:
                for attr2 in attributes:
                    val2 = solution[h + 1][attr2]
                    clues.append(("DIRECTLEFT", attr, val, attr2, val2))
            for neighbor in [h - 1, h + 1]:
                if neighbor in houses:
                    for attr2 in attributes:
                        val2 = solution[neighbor][attr2]
                        clues.append(("SIDEBYSIDE", attr, val, attr2, val2))
            for hp in houses:
                if hp > h:
                    for attr2 in attributes:
                        val2 = solution[hp][attr2]
                        clues.append(("LEFTOF", attr, val, attr2, val2))
                if hp < h:
                    for attr2 in attributes:
                        val2 = solution[hp][attr2]
                        clues.append(("RIGHTOF", attr, val, attr2, val2))

    return list(set(clues))


def pre_filter_clues(C: list[Clue]) -> list[Clue]:
    """
    OPT 1: Remove NOTAT clues that are made redundant by a FOUNDAT clue
    for the same (attr, val) — no Z3 call needed, pure O(n) filter.
    Reduces clue set by ~20-30% before the expensive minimisation loop.
    """
    foundat_vals = {(c[1], c[2]) for c in C if c[0] == "FOUNDAT"}
    return [c for c in C if not (c[0] == "NOTAT" and (c[1], c[2]) in foundat_vals)]

# ─────────────────────────────────────────────
# 3.  Z3 model
# ─────────────────────────────────────────────

def build_z3_vars(houses: list[int], attributes: Attributes):
    x = {}
    for attr in attributes:
        x[attr] = {}
        for h in houses:
            x[attr][h] = Int(f"x_{attr}_{h}")
    return x


def add_uniqueness_constraints(solver, x, houses: list[int], attributes: Attributes):
    N = len(houses)
    for attr in attributes:
        vars_ = [x[attr][h] for h in houses]
        for v in vars_:
            solver.add(v >= 0, v < N)
        solver.add(Distinct(*vars_))


def clue_to_z3(clue: Clue, x, houses: list[int], attributes: Attributes):
    tag = clue[0]
    val_idx = {attr: {v: i for i, v in enumerate(vals)}
               for attr, vals in attributes.items()}

    if tag == "FOUNDAT":
        _, attr, val, h = clue
        return x[attr][h] == val_idx[attr][val]
    elif tag == "NOTAT":
        _, attr, val, h = clue
        return x[attr][h] != val_idx[attr][val]
    elif tag == "SAMEHOUSE":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        return And(*[Implies(x[attr1][h] == idx1, x[attr2][h] == idx2) for h in houses])
    elif tag == "DIRECTLEFT":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        return And(*[Implies(x[attr1][h] == idx1, x[attr2][h + 1] == idx2) for h in houses[:-1]])
    elif tag == "SIDEBYSIDE":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        return And(*[Implies(x[attr1][h] == idx1,
                             Or(*[x[attr2][n] == idx2 for n in [h - 1, h + 1] if n in houses]))
                     for h in houses])
    elif tag == "LEFTOF":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        pairs = [(h1, h2) for h1 in houses for h2 in houses if h1 < h2]
        return Or(*[And(x[attr1][h1] == idx1, x[attr2][h2] == idx2) for h1, h2 in pairs])
    elif tag == "RIGHTOF":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        pairs = [(h1, h2) for h1 in houses for h2 in houses if h1 > h2]
        return Or(*[And(x[attr1][h1] == idx1, x[attr2][h2] == idx2) for h1, h2 in pairs])
    else:
        raise ValueError(f"Unknown clue tag: {tag}")


def count_solutions_with_solver(base_solver, x, houses, attributes, clues, max_count=2):
    """
    OPT 2: Reuse pre-built solver with push/pop instead of rebuilding
    Z3 vars and constraints from scratch on every call. 2-3x faster
    than the original count_solutions().
    """
    base_solver.push()
    for clue in clues:
        base_solver.add(clue_to_z3(clue, x, houses, attributes))

    count = 0
    while count < max_count and base_solver.check() == sat:
        count += 1
        model = base_solver.model()
        block = Or(*[x[attr][h] != model[x[attr][h]]
                     for attr in attributes for h in houses])
        base_solver.add(block)

    base_solver.pop()
    return count

# ─────────────────────────────────────────────
# 4.  Z3 Complexity Measurement  (with early exit)
# ─────────────────────────────────────────────

def measure_z3_complexity(
    clues: list[Clue],
    houses: list[int],
    attributes: Attributes,
    num_runs: int = 32,
    early_stop_std: float = 0.5,   # OPT 3: stop early if estimate is stable
) -> dict:
    """
    OPT 3: Early exit once conflict count variance drops below early_stop_std.
    Saves up to 4x on easy puzzles where Z3 is deterministic.
    """
    conflict_counts = []

    for i in range(num_runs):
        solver = Solver()
        solver.set("random_seed", random.randint(0, 100000))
        x = build_z3_vars(houses, attributes)
        add_uniqueness_constraints(solver, x, houses, attributes)
        for clue in clues:
            solver.add(clue_to_z3(clue, x, houses, attributes))
        solver.check()

        stats     = solver.statistics()
        conflicts = 0
        for key, val in stats:
            if key == "conflicts":
                conflicts = int(val)
                break
        conflict_counts.append(conflicts)

        # Early exit after min 8 runs if variance is already low
        if i >= 7:
            avg = sum(conflict_counts) / len(conflict_counts)
            std = (sum((c - avg) ** 2 for c in conflict_counts) / len(conflict_counts)) ** 0.5
            if std < early_stop_std:
                break

    N, M = len(houses), len(attributes) #N->houses, M->attributes
    factorial_N = 1
    for i in range(1, N + 1):
        factorial_N *= i
    search_space = factorial_N ** M

    if search_space < 10**3:    band = "Small"
    elif search_space < 10**6:  band = "Medium"
    elif search_space < 10**10: band = "Large"
    else:                       band = "X-Large"

    return {
        "avg_conflicts":   sum(conflict_counts) / len(conflict_counts),
        "min_conflicts":   min(conflict_counts),
        "max_conflicts":   max(conflict_counts),
        "all_conflicts":   conflict_counts,
        "search_space":    search_space,
        "complexity_band": band,
        "num_runs":        len(conflict_counts),
    }


def print_complexity_report(metrics: dict):
    print("\n=== Z3 Complexity Report ===")
    print(f"  Search space size : {metrics['search_space']:,}")
    print(f"  Complexity band   : {metrics['complexity_band']}")
    print(f"  Runs              : {metrics['num_runs']}")
    print(f"  Avg Z3 conflicts  : {metrics['avg_conflicts']:.2f}")
    print(f"  Min Z3 conflicts  : {metrics['min_conflicts']}")
    print(f"  Max Z3 conflicts  : {metrics['max_conflicts']}")
    max_bar = 40
    bar_len = min(int((metrics['avg_conflicts'] / MAX_CONFLICTS) * max_bar), max_bar)
    bar     = "█" * bar_len
    print(f"  Conflict bar      : [{bar:<{max_bar}}] {metrics['avg_conflicts']:.1f} / {MAX_CONFLICTS}")
    avg = metrics['avg_conflicts']
    if avg == 0:       verdict = "Trivial — solvable by simple forward chaining."
    elif avg < 10:     verdict = "Easy — minimal backtracking required."
    elif avg < 20:     verdict = "Moderate — some backtracking. Most LLMs still handle this."
    elif avg < 40:     verdict = "Hard — heavy backtracking. LLM accuracy drops sharply here."
    else:              verdict = "Very Hard — at this level most LLMs score near 0%."
    print(f"  Difficulty        : {verdict}")

# ─────────────────────────────────────────────
# 5.  Weighted clue sampling
# ─────────────────────────────────────────────

def sample_clue(clues: list[Clue]) -> Clue:
    weights = [CLUE_WEIGHTS.get(c[0], 1) for c in clues]
    return random.choices(clues, weights=weights, k=1)[0]

# ─────────────────────────────────────────────
# 6.  Puzzle generation  (optimised)
# ─────────────────────────────────────────────

def generate_puzzle(
    all_attributes: Attributes,
    N: int,
    M: int,
    seed: Optional[int] = None,
) -> tuple:
    if seed is not None:
        random.seed(seed)

    selected_attr_names = random.sample(list(all_attributes.keys()), M)
    attributes: Attributes = {
        a: random.sample(all_attributes[a], N)
        for a in selected_attr_names
    }
    houses   = list(range(1, N + 1))
    solution = random_solution(houses, attributes)

    C = clue_generation(solution, houses, attributes)
    C = pre_filter_clues(C)          # OPT 1: drop dominated NOTAT clues
    random.shuffle(C)

    # OPT 2: build base solver once, reuse with push/pop
    base_solver = Solver()
    x           = build_z3_vars(houses, attributes)
    add_uniqueness_constraints(base_solver, x, houses, attributes)

    i = 0
    while i < len(C):
        clue    = sample_clue(C)
        C_prime = [c for c in C if c != clue]
        if count_solutions_with_solver(base_solver, x, houses, attributes, C_prime) == 1:
            C = C_prime
        else:
            i += 1

    return solution, C, attributes

# ─────────────────────────────────────────────
# 7.  Pretty-print helpers
# ─────────────────────────────────────────────

def print_solution(solution: Solution, attributes: Attributes):
    attrs  = list(attributes.keys())
    header = f"{'House':<8}" + "".join(f"{a:<15}" for a in attrs)
    print(header)
    print("-" * len(header))
    for h in sorted(solution):
        row = f"{h:<8}" + "".join(f"{solution[h][a]:<15}" for a in attrs)
        print(row)


def clue_to_english(clue: Clue) -> str:
    tag = clue[0]
    if tag == "FOUNDAT":
        _, attr, val, h = clue
        return f"The {attr} '{val}' is in house {h}."
    elif tag == "NOTAT":
        _, attr, val, h = clue
        return f"The {attr} '{val}' is NOT in house {h}."
    elif tag == "SAMEHOUSE":
        _, a1, v1, a2, v2 = clue
        return f"The person with {a1} '{v1}' also has {a2} '{v2}'."
    elif tag == "DIRECTLEFT":
        _, a1, v1, a2, v2 = clue
        return f"The person with {a1} '{v1}' is directly left of the person with {a2} '{v2}'."
    elif tag == "SIDEBYSIDE":
        _, a1, v1, a2, v2 = clue
        return f"The person with {a1} '{v1}' is next to the person with {a2} '{v2}'."
    elif tag == "LEFTOF":
        _, a1, v1, a2, v2 = clue
        return f"The person with {a1} '{v1}' is somewhere left of the person with {a2} '{v2}'."
    elif tag == "RIGHTOF":
        _, a1, v1, a2, v2 = clue
        return f"The person with {a1} '{v1}' is somewhere right of the person with {a2} '{v2}'."
    return str(clue)

# ─────────────────────────────────────────────
# 8.  Bucket helpers
# ─────────────────────────────────────────────

def get_bucket_index(avg_conflicts: float) -> int:
    if avg_conflicts >= MAX_CONFLICTS:
        return -1
    for i in range(len(BUCKET_BOUNDARIES) - 1):
        if BUCKET_BOUNDARIES[i] <= avg_conflicts < BUCKET_BOUNDARIES[i + 1]:
            return i
    return -1

# ─────────────────────────────────────────────
# 9.  Worker
# ─────────────────────────────────────────────

def _worker(args):
    all_attributes, max_N, max_M, z3_runs, worker_seed = args
    random.seed(worker_seed)

    N = random.randint(2, max_N)
    M = random.randint(2, min(max_M, len(all_attributes)))

    if M > len(all_attributes):
        return None

    try:
        solution, clues, attributes = generate_puzzle(all_attributes, N=N, M=M)
        houses     = list(range(1, N + 1))
        metrics    = measure_z3_complexity(clues, houses, attributes, num_runs=z3_runs)
        bucket_idx = get_bucket_index(metrics["avg_conflicts"])

        if bucket_idx == -1:
            return None

        return {
            "bucket_idx": bucket_idx,
            "N":          N,
            "M":          M,
            "solution":   solution,
            "clues":      clues,
            "attributes": attributes,
            "metrics":    metrics,
        }

    except Exception as e:
        print(f"  Worker error: N={N} M={M} | ERROR: {e}", flush=True)
        return None

# ─────────────────────────────────────────────
# 10.  Incremental file writing
# ─────────────────────────────────────────────

def _init_output_file(file_path: str, k: int, n_buckets: int):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  ZEBRALOGIC PUZZLE DATASET\n")
        f.write(f"  Buckets : {n_buckets}  ({', '.join(BUCKET_LABELS)})\n")
        f.write(f"  k       : {k} puzzles per bucket\n")
        f.write(f"  Total   : {k * n_buckets} puzzles (target)\n")
        f.write(f"  Max conflicts cap : {MAX_CONFLICTS}\n")
        f.write("=" * 65 + "\n\n")


def _write_puzzle_to_file(file_path: str, puzzle: dict, puzzle_number: int):
    """Append a single puzzle immediately after it is accepted — no lock needed
    since this is always called from the single-threaded main process."""
    N          = puzzle["N"]
    M          = puzzle["M"]
    solution   = puzzle["solution"]
    clues      = puzzle["clues"]
    attributes = puzzle["attributes"]
    metrics    = puzzle["metrics"]
    bucket_idx = puzzle["bucket_idx"]
    label      = BUCKET_LABELS[bucket_idx]

    with open(file_path, "a") as f:
        f.write(f"Puzzle {puzzle_number}  [Bucket: {label}]  [N={N} houses, M={M} attributes]\n")
        f.write("-" * 65 + "\n")

        f.write("Complexity Metrics:\n")
        f.write(f"  Avg Z3 conflicts  : {metrics['avg_conflicts']:.2f}\n")
        f.write(f"  Min Z3 conflicts  : {metrics['min_conflicts']}\n")
        f.write(f"  Max Z3 conflicts  : {metrics['max_conflicts']}\n")
        f.write(f"  Search space      : {metrics['search_space']:,}\n")
        f.write(f"  Complexity band   : {metrics['complexity_band']}\n\n")

        f.write("Background:\n")
        f.write(f"  There are {N} houses, numbered 1 to {N} from left to right.\n")
        f.write("  Each house has a unique value for each attribute:\n")
        for attr, vals in attributes.items():
            f.write(f"    - {attr}: {', '.join(vals)}\n")
        f.write("\n")

        f.write(f"Clues ({len(clues)} total):\n")
        for clue_num, clue in enumerate(clues, 1):
            f.write(f"  {clue_num:2d}. {clue_to_english(clue)}\n")
        f.write("\n")

        f.write("Solution:\n")
        attrs  = list(attributes.keys())
        header = f"  {'House':<8}" + "".join(f"{a:<15}" for a in attrs)
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        for h in sorted(solution):
            row = f"  {h:<8}" + "".join(f"{solution[h][a]:<15}" for a in attrs)
            f.write(row + "\n")

        f.write("\n" + "=" * 65 + "\n\n")

# ─────────────────────────────────────────────
# 11.  Histogram generation
# ─────────────────────────────────────────────
#N->houses, M->attributes
def generate_histogram_data(
    all_attributes,
    k: int              = 5,
    max_N: int          = 3,
    max_M: int          = 3,
    z3_runs: int        = 32,
    seed: Optional[int] = None,
    num_workers: int    = 5,
    file_path: str      = "data_creation/zebralogic_puzzles.txt",
) -> dict:
    if seed is not None:
        random.seed(seed)

    print(f"Generating histogram: k={k}/bucket, max_N={max_N}, max_M={max_M}, "
          f"z3_runs={z3_runs}, num_workers={num_workers}\n", flush=True)

    n_buckets    = len(BUCKET_LABELS)
    buckets      = [[] for _ in range(n_buckets)]
    attempts     = 0
    last_report  = 0
    puzzle_count = 0

    _init_output_file(file_path, k, n_buckets)
    t0 = time.time()

    with Pool(processes=num_workers) as pool:
        while not all(len(b) >= k for b in buckets):
            batch_args = [
                (all_attributes, max_N, max_M, z3_runs, random.randint(0, 2**31))
                for _ in range(num_workers)
            ]

            results   = pool.map(_worker, batch_args)
            attempts += num_workers

            if attempts - last_report >= 100:
                last_report = attempts
                filled  = sum(1 for b in buckets if len(b) >= k)
                waiting = [BUCKET_LABELS[i] for i, b in enumerate(buckets) if len(b) < k]
                print(f"  Attempt {attempts:4d} | Filled: {filled}/{n_buckets} | "
                      f"Waiting: {waiting}", flush=True)

            for result in results:
                if result is None:
                    continue

                bucket_idx = result["bucket_idx"]

                if len(buckets[bucket_idx]) < k:
                    buckets[bucket_idx].append(result)
                    puzzle_count += 1
                    filled = sum(1 for b in buckets if len(b) >= k)

                    _write_puzzle_to_file(file_path, result, puzzle_count)

                    print(f"  Attempt {attempts:4d} | N={result['N']} M={result['M']} | "
                          f"avg_conflicts={result['metrics']['avg_conflicts']:5.2f} | "
                          f"bucket='{BUCKET_LABELS[bucket_idx]}' "
                          f"({len(buckets[bucket_idx])}/{k}) | "
                          f"filled: {filled}/{n_buckets} | "
                          f"saved: #{puzzle_count}", flush=True)

                    # ── Time report every 20 puzzles ──────────────────
                    if puzzle_count % 20 == 0:
                        elapsed   = time.time() - t0
                        rate      = puzzle_count / elapsed
                        remaining = (k * n_buckets - puzzle_count) / rate if rate > 0 else float("inf")
                        print(f"\n  ── Checkpoint: {puzzle_count}/{k * n_buckets} puzzles"
                              f" | Elapsed: {elapsed:.1f}s"
                              f" | Rate: {rate:.2f} puzzles/s"
                              f" | ETA: {remaining:.0f}s ──\n", flush=True)

    filled_buckets = sum(1 for b in buckets if len(b) >= k)
    print(f"\nDone! {filled_buckets}/{n_buckets} buckets filled after {attempts} attempts.")
    print(f"Puzzles written to: {file_path}")
    return {
        "buckets":        buckets,
        "attempts":       attempts,
        "filled_buckets": filled_buckets,
        "k":              k,
    }

# ─────────────────────────────────────────────
# 12.  Plotting & summary
# ─────────────────────────────────────────────

def plot_histogram(result: dict, save_path: Optional[str] = None):
    buckets   = result["buckets"]
    k         = result["k"]
    n_buckets = len(BUCKET_LABELS)
    counts    = [len(b) for b in buckets]
    all_conflicts_per_bucket = [[p["metrics"]["avg_conflicts"] for p in b] for b in buckets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"ZebraLogic Difficulty Histogram — max conflicts: {MAX_CONFLICTS}, k={k} per bucket",
        fontsize=13, fontweight="bold"
    )

    ax   = axes[0]
    x    = range(n_buckets)
    bars = ax.bar(x, counts, color=BUCKET_COLORS, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axhline(k, color="black", linestyle="--", linewidth=1.2, label=f"Target k={k}", zorder=3)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Average Z3 Conflicts (difficulty bucket)", fontsize=11)
    ax.set_ylabel("Number of Puzzles", fontsize=11)
    ax.set_title("Puzzle Count per Difficulty Bucket", fontsize=12)
    ax.set_ylim(0, max(counts + [k]) * 1.2 + 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4, zorder=1)

    ax2 = axes[1]
    for i, conflicts in enumerate(all_conflicts_per_bucket):
        if conflicts:
            jitter = [i + random.uniform(-0.2, 0.2) for _ in conflicts]
            ax2.scatter(jitter, conflicts, color=BUCKET_COLORS[i], alpha=0.75, s=45,
                        edgecolors="white", linewidths=0.4, zorder=2)
    for boundary in BUCKET_BOUNDARIES[1:-1]:
        ax2.axhline(boundary, color="grey", linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
    ax2.axhline(MAX_CONFLICTS, color="black", linestyle="--", linewidth=1.0,
                alpha=0.5, label=f"Cap = {MAX_CONFLICTS}", zorder=3)
    ax2.set_xticks(list(range(n_buckets)))
    ax2.set_xticklabels(BUCKET_LABELS, rotation=20, ha="right", fontsize=10)
    ax2.set_xlabel("Difficulty Bucket", fontsize=11)
    ax2.set_ylabel("Individual Avg Z3 Conflicts", fontsize=11)
    ax2.set_title("Per-Puzzle Conflict Scores within Each Bucket", fontsize=12)
    ax2.set_ylim(0, MAX_CONFLICTS + 2)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=1)

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Easy (0–10)"),
        mpatches.Patch(color="#FFC107", label="Moderate (10–20, LLMs start failing)"),
        mpatches.Patch(color="#FF5722", label="Hard (20–30)"),
        mpatches.Patch(color="#F44336", label="Very Hard (30–40)"),
        mpatches.Patch(color="#9C27B0", label="Extreme (40–50)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nHistogram saved to: {save_path}")
    plt.show()


def print_histogram_summary(result: dict):
    buckets        = result["buckets"]
    attempts       = result["attempts"]
    filled_buckets = result["filled_buckets"]
    k              = result["k"]

    print("\n" + "=" * 65)
    print(f"  HISTOGRAM SUMMARY  (k={k} per bucket, max conflicts={MAX_CONFLICTS})")
    print("=" * 65)
    print(f"  Total attempts          : {attempts}")
    print(f"  Buckets filled (>= k)   : {filled_buckets}/{len(BUCKET_LABELS)}")
    print()
    print(f"  {'Bucket':<10} {'Count':>6}  {'Avg Conflicts':>14}  "
          f"{'N range':>10}  {'M range':>10}  {'Status':>6}")
    print(f"  {'-'*10} {'-'*6}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*6}")

    for label, bucket in zip(BUCKET_LABELS, buckets):
        if bucket:
            avg_c   = sum(p["metrics"]["avg_conflicts"] for p in bucket) / len(bucket)
            n_vals  = [p["N"] for p in bucket]
            m_vals  = [p["M"] for p in bucket]
            n_range = f"{min(n_vals)}–{max(n_vals)}"
            m_range = f"{min(m_vals)}–{max(m_vals)}"
            status  = "✓" if len(bucket) >= k else "✗"
            print(f"  {label:<10} {len(bucket):>6}  {avg_c:>14.2f}  "
                  f"{n_range:>10}  {m_range:>10}  {status:>6}")
        else:
            print(f"  {label:<10} {0:>6}  {'—':>14}  {'—':>10}  {'—':>10}  {'✗':>6}")
    print("=" * 65)

# ─────────────────────────────────────────────
# 13.  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=100)
    parser.add_argument("--k",           type=int, default=200)   # 200 × 5 = 1000
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    ALL_ATTRIBUTES: Attributes = {
        "Name":        ["Eric", "Peter", "Arnold", "Sara", "Tom", "Lucy"],
        "Drink":       ["milk", "water", "tea", "juice", "coffee", "soda"],
        "Hobby":       ["photography", "cooking", "gardening", "reading", "gaming", "painting"],
        "Color":       ["red", "blue", "green", "yellow", "white", "purple"],
        "Nationality": ["British", "Swedish", "Danish", "Norwegian", "German", "Finnish"],
    }

    print(f"Generating histogram — 5 buckets (0–{MAX_CONFLICTS} conflicts), "
          f"k={args.k} each ({args.k * 4} total), {args.num_workers} workers ...\n")

    t0 = time.time()

    result = generate_histogram_data(
        all_attributes = ALL_ATTRIBUTES,
        k              = args.k, #number of puzzles per bucket
        max_N          = 4, #number of houses
        max_M          = 4, #number of attributes (Name, Drink, Hobby, Color, Nationality)
        z3_runs        = 32,
        seed           = args.seed,
        num_workers    = args.num_workers,
        file_path      = "data_creation/zebralogic_puzzles_800_max_difficulty_15_M4_N4.txt",
    )

    print(f"Generated 1000 puzzles with max 15 conflicts, 4 attributes, 4 houses")
    t_parallel = time.time() - t0
    print(f"\n  Total time : {t_parallel:.1f}s")

    print_histogram_summary(result)
    plot_histogram(result, save_path="data_creation/zebralogic_histogram_800_max_difficulty_15_M4_N4.png")
