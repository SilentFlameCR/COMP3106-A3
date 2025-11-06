# test.py — grade-style runner for COMP3106-A3
# Usage:
#   python3 test.py           # run all Example*
#   python3 test.py 0         # run only Example0
#   python3 test.py 0 1 3     # run specific examples

import csv, glob, os, sys, math
from assignment3 import td_qlearning  # your implementation  # noqa

ABS_TOL = 1e-3     # tolerance for Q-values
REL_TOL = 1e-5

def _norm_state(s: str) -> str:
    # normalize possible Unicode minus '−' to '-' (some editors export it)
    return s.replace('−', '-').strip()

def _is_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)

def run_one(example_dir: str) -> None:
    trials_dir = os.path.join(example_dir, "Trials")
    name = os.path.basename(example_dir)
    print(f"\n===== {name} =====")

    if not os.path.isdir(trials_dir):
        print(f"[skip] No Trials/ in {example_dir}")
        return

    # Train model
    model = td_qlearning(trials_dir)

    # ---- Q-VALUE TESTS ----
    qcsv = os.path.join(example_dir, "qvalue_tests.csv")
    if os.path.isfile(qcsv):
        print("\n--- Q-value tests ---")
        with open(qcsv, "r", newline="") as f:
            r = csv.reader(f)
            row_idx = 0
            for row in r:
                row_idx += 1
                if not row:
                    continue
                # Formats supported:
                #   state, action
                #   state, action, expected_q
                state = _norm_state(row[0])
                action = int(row[1])
                q = model.qvalue(state, action)
                if len(row) >= 3 and row[2].strip() != "":
                    try:
                        expected = float(row[2])
                        ok = _is_close(q, expected)
                        verdict = "PASS" if ok else "FAIL"
                        print(f"[{verdict}] Q({state}, {action}) = {q:.6f}  (expected {expected:.6f})")
                    except ValueError:
                        print(f"[WARN] line {row_idx}: expected value not a float -> {row[2]!r}")
                        print(f"      Q({state}, {action}) = {q:.6f}")
                else:
                    print(f"Q({state}, {action}) = {q:.6f}")
    else:
        print("[info] No qvalue_tests.csv found")

    # ---- POLICY TESTS ----
    pcsv = os.path.join(example_dir, "policy_tests.csv")
    if os.path.isfile(pcsv):
        print("\n--- Policy tests ---")
        with open(pcsv, "r", newline="") as f:
            r = csv.reader(f)
            row_idx = 0
            for row in r:
                row_idx += 1
                if not row:
                    continue
                # Formats supported:
                #   state
                #   state, expected_action
                state = _norm_state(row[0])
                a = model.policy(state)
                if len(row) >= 2 and row[1].strip() != "":
                    try:
                        expected_a = int(row[1])
                        verdict = "PASS" if a == expected_a else "FAIL"
                        print(f"[{verdict}] policy({state}) = {a}  (expected {expected_a})")
                    except ValueError:
                        print(f"[WARN] line {row_idx}: expected action not an int -> {row[1]!r}")
                        print(f"      policy({state}) = {a}")
                else:
                    print(f"policy({state}) = {a}")
    else:
        print("[info] No policy_tests.csv found")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    examples = sorted(glob.glob(os.path.join(base, "Examples", "Example*")))
    # Filter by CLI args if provided (e.g., 0 1 3)
    if len(sys.argv) > 1:
        wanted = {f"Example{arg}" for arg in sys.argv[1:]}
        examples = [e for e in examples if os.path.basename(e) in wanted]
    if not examples:
        print("No Examples/Example*/ directories found or matched.")
        return
    for ex in examples:
        run_one(ex)

if __name__ == "__main__":
    main()
