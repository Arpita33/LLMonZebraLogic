import random
from itertools import permutations
from typing import Optional
from z3 import *

# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────

Attributes = dict[str, list[str]]          # e.g. {"Name": ["Eric","Peter","Arnold"], "Drink": [...]}
Solution   = dict[int, dict[str, str]]     # house_index -> {attr -> value}
Clue       = tuple                         # internal representation – see ClueGeneration


# ─────────────────────────────────────────────
# 1.  Random solution grid
# ─────────────────────────────────────────────

def random_solution(houses: list[int], attributes: Attributes) -> Solution:
    """Randomly assign values to every (house, attribute) cell."""
    solution: Solution = {h: {} for h in houses}
    for attr, values in attributes.items():
        shuffled = values[:]
        random.shuffle(shuffled)
        for h, v in zip(houses, shuffled):
            solution[h][attr] = v
    return solution


# ─────────────────────────────────────────────
# 2.  Clue types & generation
# ─────────────────────────────────────────────

def clue_generation(solution: Solution, houses: list[int], attributes: Attributes) -> list[Clue]:
    """
    Generate ALL valid clues that are consistent with *solution*.
    Each clue is a tuple whose first element is a string tag.
    """
    clues: list[Clue] = []
    N = len(houses)

    for attr, values in attributes.items():
        for val in values:
            # Which house holds this value?
            h = next(hh for hh in houses if solution[hh][attr] == val)

            # FOUNDAT  – "X is in house h"
            clues.append(("FOUNDAT", attr, val, h))

            # NOTAT    – "X is NOT in house h'" for all h' != h
            for hp in houses:
                if hp != h:
                    clues.append(("NOTAT", attr, val, hp))

            # SAMEHOUSE – correlate with every other attribute/value in same house
            for attr2, values2 in attributes.items():
                if attr2 == attr:
                    continue
                val2 = solution[h][attr2]
                clues.append(("SAMEHOUSE", attr, val, attr2, val2))

            # DIRECTLEFT – "X is directly left of Y"
            if h < N:
                for attr2, values2 in attributes.items():
                    val2 = solution[h + 1][attr2]
                    clues.append(("DIRECTLEFT", attr, val, attr2, val2))

            # SIDEBYSIDE – symmetric neighbour
            for neighbor in [h - 1, h + 1]:
                if neighbor in houses:
                    for attr2, values2 in attributes.items():
                        val2 = solution[neighbor][attr2]
                        clues.append(("SIDEBYSIDE", attr, val, attr2, val2))

            # LEFTOF / RIGHTOF
            for hp in houses:
                if hp > h:
                    for attr2 in attributes:
                        val2 = solution[hp][attr2]
                        clues.append(("LEFTOF", attr, val, attr2, val2))
                if hp < h:
                    for attr2 in attributes:
                        val2 = solution[hp][attr2]
                        clues.append(("RIGHTOF", attr, val, attr2, val2))

    # Deduplicate
    return list(set(clues))


# ─────────────────────────────────────────────
# 3.  SAT/SMT model (Z3) for counting solutions
# ─────────────────────────────────────────────

def build_z3_vars(houses: list[int], attributes: Attributes):
    """Return a dict x[attr][house] = Int Z3 variable (index into values list)."""
    x = {}
    for attr, values in attributes.items():
        x[attr] = {}
        for h in houses:
            x[attr][h] = Int(f"x_{attr}_{h}")
    return x


def add_uniqueness_constraints(solver: Solver, x, houses: list[int], attributes: Attributes):
    N = len(houses)
    for attr, values in attributes.items():
        vars_ = [x[attr][h] for h in houses]
        # domain
        for v in vars_:
            solver.add(v >= 0, v < N)
        # all-different
        solver.add(Distinct(*vars_))


def clue_to_z3(clue: Clue, x, houses: list[int], attributes: Attributes) -> BoolRef:
    """Translate one clue tuple to a Z3 Boolean expression."""
    tag = clue[0]
    val_idx = {attr: {v: i for i, v in enumerate(vals)} for attr, vals in attributes.items()}

    if tag == "FOUNDAT":
        _, attr, val, h = clue
        return x[attr][h] == val_idx[attr][val]

    elif tag == "NOTAT":
        _, attr, val, h = clue
        return x[attr][h] != val_idx[attr][val]

    elif tag == "SAMEHOUSE":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        constraints = []
        for h in houses:
            constraints.append(
                Implies(x[attr1][h] == idx1, x[attr2][h] == idx2)
            )
        return And(*constraints)

    elif tag == "DIRECTLEFT":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        constraints = []
        for h in houses[:-1]:
            constraints.append(
                Implies(x[attr1][h] == idx1, x[attr2][h + 1] == idx2)
            )
        return And(*constraints)

    elif tag == "SIDEBYSIDE":
        _, attr1, val1, attr2, val2 = clue
        idx1, idx2 = val_idx[attr1][val1], val_idx[attr2][val2]
        constraints = []
        for h in houses:
            neighbors = [hp for hp in [h - 1, h + 1] if hp in houses]
            if neighbors:
                constraints.append(
                    Implies(x[attr1][h] == idx1,
                            Or(*[x[attr2][n] == idx2 for n in neighbors]))
                )
        return And(*constraints)

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


def count_solutions(clues: list[Clue], houses: list[int], attributes: Attributes,
                    max_count: int = 2) -> int:
    """Return the number of distinct solutions (up to max_count) under the given clues."""
    solver = Solver()
    x = build_z3_vars(houses, attributes)
    add_uniqueness_constraints(solver, x, houses, attributes)

    for clue in clues:
        solver.add(clue_to_z3(clue, x, houses, attributes))

    count = 0
    while count < max_count and solver.check() == sat:
        count += 1
        model = solver.model()
        # Block this solution
        block = Or(*[x[attr][h] != model[x[attr][h]]
                     for attr in attributes for h in houses])
        solver.add(block)

    return count


# ─────────────────────────────────────────────
# 4.  Weighted clue sampling
# ─────────────────────────────────────────────

CLUE_WEIGHTS = {
    "FOUNDAT":    5,   # simplest  → higher weight → removed first
    "NOTAT":      4,
    "SAMEHOUSE":  3,
    "DIRECTLEFT": 2,
    "SIDEBYSIDE": 2,
    "LEFTOF":     1,
    "RIGHTOF":    1,
}

def sample_clue(clues: list[Clue]) -> Clue:
    weights = [CLUE_WEIGHTS.get(c[0], 1) for c in clues]
    return random.choices(clues, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# 5.  Algorithm 1 – ZebraLogic Puzzle Generation
# ─────────────────────────────────────────────

def generate_puzzle(
    all_attributes: Attributes,
    N: int,
    M: int,
    seed: Optional[int] = None,
) -> tuple[Solution, list[Clue]]:
    """
    Algorithm 1 from the ZebraLogic paper.

    Parameters
    ----------
    all_attributes : dict mapping attribute name -> list of possible values
                     (each list must have at least N values)
    N              : number of houses
    M              : number of attributes to sample
    seed           : optional random seed for reproducibility

    Returns
    -------
    (solution, minimal_clue_set)
    """
    if seed is not None:
        random.seed(seed)

    # Line 1 – sample M attributes
    selected_attr_names = random.sample(list(all_attributes.keys()), M)
    attributes: Attributes = {
        a: random.sample(all_attributes[a], N)   # pick exactly N values
        for a in selected_attr_names
    }

    houses = list(range(1, N + 1))

    # Line 2 – random solution grid S
    solution = random_solution(houses, attributes)

    # Line 3 – generate all valid clues
    C = clue_generation(solution, houses, attributes)

    # Lines 4-11 – iterative minimisation
    random.shuffle(C)
    i = 0
    while i < len(C):
        clue = sample_clue(C)          # line 5
        C_prime = [c for c in C if c != clue]   # line 6

        # Line 7 – does C' still uniquely determine S?
        if count_solutions(C_prime, houses, attributes) == 1:
            C = C_prime                # line 8 – permanently remove
            # don't advance i; the list shrank
        else:
            i += 1                     # keep clue, try next

    return solution, C                 # line 12


# ─────────────────────────────────────────────
# 6.  Pretty-print helpers
# ─────────────────────────────────────────────

def print_solution(solution: Solution, attributes: Attributes):
    attrs = list(attributes.keys())
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
# 7.  Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ALL_ATTRIBUTES: Attributes = {
        "Name":       ["Eric", "Peter", "Arnold", "Sara", "Tom", "Lucy"],
        "Drink":      ["milk", "water", "tea", "juice", "coffee", "soda"],
        "Hobby":      ["photography", "cooking", "gardening", "reading", "gaming", "painting"],
        "Color":      ["red", "blue", "green", "yellow", "white", "purple"],
        "Nationality":["British", "Swedish", "Danish", "Norwegian", "German", "Finnish"],
    }

    print("Generating a 3×3 ZebraLogic puzzle ...\n")
    solution, clues = generate_puzzle(ALL_ATTRIBUTES, N=3, M=3, seed=42)

    print("=== Solution ===")
    attrs = {a: ALL_ATTRIBUTES[a] for a in list(solution[1].keys())}
    print_solution(solution, attrs)

    print(f"\n=== Minimal Clue Set ({len(clues)} clues) ===")
    for i, clue in enumerate(clues, 1):
        print(f"  {i}. {clue_to_english(clue)}")