# LLMonZebraLogic — ZebraLogic Puzzle Generator

This repository contains a puzzle generator and dataset creation tool for Zebra / logic grid puzzles ("Zebra Logic"). The generator produces puzzles together with Z3-based complexity metrics that estimate how difficult a puzzle is for a solver (measured by average Z3 conflicts). It can build a stratified dataset of puzzles binned into difficulty buckets and save human-readable puzzle files and a difficulty histogram.

This README was generated from the current code in `Optimized_ZL_Parallel.py` and documents usage, configuration, and implementation notes.

---

## Key features

- Generates Zebra-style logic puzzles (houses, attributes, clues, solutions).
- Uses the Z3 SMT solver to measure puzzle solving difficulty (counts solver conflicts over repeated runs).
- Produces puzzles stratified into difficulty buckets (based on average Z3 conflict counts).
- Parallel puzzle search using Python multiprocessing to fill buckets quickly.
- Saves each accepted puzzle (background, clues, and solution) to a text file incrementally so progress is not lost.
- Produces a histogram plot summarizing puzzles per difficulty bucket and per-puzzle conflict scores.

---

## Files of interest

- `Optimized_ZL_Parallel.py` — main generator with optimizations:
  - Pre-filters redundant `NOTAT` clues.
  - Reuses a base Z3 solver with push/pop to avoid rebuilding constraints repeatedly.
  - Early-stops Z3 complexity sampling when the conflict variance stabilizes.
  - Parallel worker pool to generate candidate puzzles concurrently.

- `data_creation/` (output path used by the script) — by default puzzles are appended to a human-readable file such as `data_creation/zebralogic_puzzles_800_max_difficulty_15_M4_N4.txt` and a histogram image is saved when plotting.

---

## Dependencies

- Python 3.8+ (tested with 3.8–3.11)
- z3-solver (Z3 Python bindings)
- matplotlib

Install dependencies with pip:

pip install z3-solver matplotlib

Note: Z3 must be available to the Python package. On some systems installing `z3-solver` via pip is sufficient; consult Z3 documentation if you encounter installation issues.

---

## How it works (high-level)

1. Select M attribute categories (e.g., Name, Drink, Color) and N houses.
2. Randomly generate a complete solution grid where each house has a unique value for each attribute.
3. Derive a large pool of logically implied clues from the solution (FOUNDAT, NOTAT, SAMEHOUSE, DIRECTLEFT, SIDEBYSIDE, LEFTOF, RIGHTOF).
4. Optionally pre-filter trivial/ redundant NOTAT clues that are implied by FOUNDAT for the same (attribute, value).
5. Iteratively attempt to remove clues while keeping the puzzle uniquely solvable. Clue removal decisions are sampled with weights (some clue types are preferred) to create concise puzzles.
6. For candidate puzzles, run multiple Z3 solver instances (with different random seeds) to measure average solver conflicts; this is used as a proxy difficulty metric.
7. Accept puzzles whose average conflict score falls into one of the predefined difficulty buckets, append immediately to the output file, and continue until `k` puzzles are collected per bucket.

---

## Difficulty buckets and metrics

- Difficulty is measured by the average number of Z3 "conflicts" observed while solving the puzzle across repeated runs.
- Bucket boundaries and labels are configurable via constants in the code. Default in `Optimized_ZL_Parallel.py`:
  - `BUCKET_BOUNDARIES = [0, 3, 6, 9, 12, 15]`
  - `BUCKET_LABELS = ["0–3", "3–6", "6–9", "9–12", "12–15"]`
  - `MAX_CONFLICTS = 20` — used as a cap/visual reference in plots.

These settings map average conflict scores into buckets (easy → hard). The script also reports search space size, min/ max conflicts observed, and a short human verdict (Trivial, Easy, Moderate, Hard, Very Hard).

---

## Usage

Run the main script from the repository root. The entry point supports a few CLI flags:

python Optimized_ZL_Parallel.py --num_workers 100 --k 200 --seed 42

Important runtime configuration occurs inside the script when calling `generate_histogram_data()` near the `if __name__ == '__main__'` block. The key parameters of `generate_histogram_data()` are:

- all_attributes: dict of attribute_name -> list of values (the full pool from which attributes are sampled).
- k: number of puzzles per bucket to collect (default example sets `k=200`).
- max_N: maximum number of houses (N) allowed for generated puzzles.
- max_M: maximum number of attributes (M) allowed for generated puzzles.
- z3_runs: number of independent Z3 runs to measure average conflict (default 32 in code).
- seed: random seed for reproducible generation.
- num_workers: number of parallel worker processes to use.
- file_path: path to the output text file where puzzles are appended.

Example: smaller test run with fewer workers and fewer puzzles per bucket (good for testing locally):

python -c "from Optimized_ZL_Parallel import generate_histogram_data, ALL_ATTRIBUTES
result = generate_histogram_data(all_attributes=ALL_ATTRIBUTES, k=2, max_N=4, max_M=4, z3_runs=8, seed=123, num_workers=4, file_path='data_creation/test_puzzles.txt')"

Or call the module directly with the bundled CLI, and edit defaults at the bottom of the file if needed.

---

## Output

- A human-readable text file (`file_path`) receives puzzles as they are accepted. Each entry contains:
  - Complexity metrics (avg/min/max conflicts, search space)
  - Background / attributes listing
  - Numbered clues in plain English
  - A tabular solution grid

- Optionally, a histogram plot can be produced by calling `plot_histogram(result, save_path=...)`, which will save a PNG and show an interactive plot.

---

## Performance & Optimizations in the code

- Pre-filtering of NOTAT clues that are redundant with FOUNDAT reduces the number of Z3 constraints early.
- Reusing a single base Z3 solver with push/pop (`count_solutions_with_solver`) avoids rebuilding identical common constraints repeatedly.
- Early-stopping the Z3 conflict-sampling loop when measured variance stabilizes saves runs on deterministic/easy cases.
- Worker processes perform candidate generation in parallel and return puzzles that match a difficulty bucket.
- Each accepted puzzle is appended to disk immediately to preserve progress even if the job is interrupted.

---

## Troubleshooting

- If Z3 is not installed or the Python binding fails to import, ensure `z3-solver` is installed and compatible with your environment. On some systems you may need to install a platform-specific wheel from the Z3 project.
- Long runs and CPU-bound behavior are expected when `k` and `num_workers` are large. Use fewer workers for local testing.
- If the generator stalls trying to fill harder buckets, try increasing search time (by increasing the number of attempts or running for longer) or adjust `BUCKET_BOUNDARIES` to match your target difficulty distribution.

---

## Notes & Next steps

- The current default attributes and constants are a small example in the script. You can extend `ALL_ATTRIBUTES` for richer datasets.
- Consider adding CLI flags for most parameters (k, max_N, max_M, z3_runs, output file path) so behavior can be tuned without editing the file.
- Additional output formats (JSON, CSV) could be added in `_write_puzzle_to_file` if downstream programs need structured data.

---

## License

This repository does not currently include a license file. Add a LICENSE if you want to grant explicit reuse permissions.

---

If you'd like, I can:
- Add README badges, examples, or a short quick-start script.
- Add CLI flags to make parameters configurable without editing the script.
- Export produced puzzles in JSON for programmatic consumption.

