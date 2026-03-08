pip install z3-solver
How it works:
```

By this point, `C` is a *minimal* clue set — no single clue can be removed without making the puzzle ambiguous. Combined with the background uniqueness constraints, this gives a puzzle with exactly one valid solution: the original `solution` grid.

---

**Summary of the flow:**

Random solution grid
        ↓
Generate ALL true clues (over-determined)
        ↓
Repeatedly attempt to remove clues (weighted toward simpler ones)
        ↓
Keep removal only if Z3 confirms solution is still unique
        ↓
Stop when no more clues can be removed → minimal puzzle
```