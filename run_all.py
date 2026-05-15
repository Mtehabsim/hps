"""
HPS Sentinel — Run All Tests
══════════════════════════════
Executes all 4 validation tests in sequence and produces a summary report.

Usage:
  python run_all.py [--tests 1,2,3,4]  # run specific tests (default: all)
  python run_all.py --tests 1,3        # run only tests 1 and 3

Output:
  results/summary_report.json   — machine-readable summary
  results/summary_report.txt    — human-readable summary
  plots/                        — all visualisations

Tip: On first run, start with --tests 3 (TRACED space) which is the
fastest and most informative about whether the signal exists.
"""

import sys, os, argparse, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def _load_result(filename: str) -> dict | None:
    path = os.path.join(config.RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_test(name: str, module_path: str, test_num: int) -> dict:
    """Import and run a test module's main() function."""
    import importlib.util
    spec   = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    t0 = time.time()
    try:
        module.main()
        elapsed = time.time() - t0
        return {"status": "PASS", "elapsed_s": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[run_all] ✗ Test {test_num} FAILED: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e), "elapsed_s": round(elapsed, 1)}


def generate_summary(test_results: dict) -> str:
    """Build a human-readable summary from saved result files."""
    lines = ["", "=" * 65, "  HPS SENTINEL — VALIDATION SUMMARY", "=" * 65, ""]

    # ── Test 1 ──
    if "1" in test_results:
        lines.append("TEST 1 — Gromov δ-Hyperbolicity")
        lines.append(f"  Status: {test_results['1']['status']}")
        r = _load_result("test1_gromov_delta.json")
        if r and "layer_results" in r:
            for cat, layers in r["layer_results"].items():
                deltas = [v["max_delta"] for v in layers.values()]
                if deltas:
                    avg = sum(deltas) / len(deltas)
                    verdict = "HYPERBOLIC" if avg < 1.0 else "MILDLY" if avg < 2.0 else "EUCLIDEAN"
                    lines.append(f"  {cat:15s}: avg max_δ = {avg:.3f}  [{verdict}]")
        lines.append("")

    # ── Test 2 ──
    if "2" in test_results:
        lines.append("TEST 2 — Hierarchy Radial Structure")
        lines.append(f"  Status: {test_results['2']['status']}")
        r = _load_result("test2_hierarchy.json")
        if r:
            po = r.get("pair_ordering", {})
            acc = po.get("accuracy", float("nan"))
            lines.append(f"  Ordering accuracy: {po.get('n_correct','?')}/{po.get('n_total','?')} ({acc:.0%})")
            lines.append(f"  Verdict: {'PASS ✓ (≥70%)' if acc >= 0.7 else 'FAIL ✗'}")
            for cat, stats in r.get("category_radii", {}).items():
                lines.append(f"  {cat:15s}: mean_radius = {stats['mean']:.4f}")
        lines.append("")

    # ── Test 3 ──
    if "3" in test_results:
        lines.append("TEST 3 — TRACED Space & Trajectory Kinematics")
        lines.append(f"  Status: {test_results['3']['status']}")
        r = _load_result("test3_traced_space.json")
        if r:
            au = r.get("auroc", {})
            eu  = au.get("euclidean",  float("nan"))
            hyp = au.get("hyperbolic", float("nan"))
            lines.append(f"  Euclidean  AUROC: {eu:.3f}")
            lines.append(f"  Hyperbolic AUROC: {hyp:.3f}")
            if eu == eu and hyp == hyp:  # not nan
                lines.append(f"  Δ AUROC: {hyp-eu:+.3f}  "
                              f"({'Hyperbolic adds signal ✓' if hyp > eu else 'No benefit from geometry ✗'})")
        lines.append("")

    # ── Test 4 ──
    if "4" in test_results:
        lines.append("TEST 4 — Probe Baseline Comparison")
        lines.append(f"  Status: {test_results['4']['status']}")
        r = _load_result("test4_baseline_comparison.json")
        if r:
            probes = r.get("probes", {})
            for name, res in probes.items():
                lines.append(f"  {name:12s}: AUROC={res.get('auroc', float('nan')):.3f}  "
                              f"FPR@95={res.get('fpr_at_95', float('nan')):.3f}  "
                              f"F1={res.get('f1', float('nan')):.3f}")
            du = r.get("dual_use_fpr", {})
            if du:
                lines.append("  Dual-use FPR:")
                for name, fpr in du.items():
                    lines.append(f"    {name}: {fpr:.3f} "
                                 f"({'GOOD' if fpr < 0.15 else 'HIGH'})")
            lines.append(f"  {r.get('verdict', '')}")
        lines.append("")

    # ── Overall ──
    lines += ["=" * 65, "OVERALL VERDICT", "─" * 65]
    r3 = _load_result("test3_traced_space.json")
    r4 = _load_result("test4_baseline_comparison.json")

    signals = []
    if r3:
        au = r3.get("auroc", {})
        if au.get("hyperbolic", 0) > 0.7:
            signals.append("✓ Hyperbolic trajectory signal present (AUROC>0.7)")
        else:
            signals.append("✗ Trajectory signal weak — review layer selection")
    if r4:
        delta = r4.get("delta_auroc", 0)
        if delta > 0.03:
            signals.append("✓ Hyperbolic geometry adds meaningful discriminative power")
        else:
            signals.append("✗ Geometry not clearly adding value over Euclidean baseline")

    r2 = _load_result("test2_hierarchy.json")
    if r2:
        acc = r2.get("pair_ordering", {}).get("accuracy", 0)
        if acc >= 0.7:
            signals.append("✓ Poincaré ball encodes abstraction hierarchy correctly")
        else:
            signals.append("✗ Hierarchy not faithfully encoded — radial coordinate unreliable")

    for s in signals:
        lines.append(f"  {s}")

    n_pass = sum("✓" in s for s in signals)
    n_total = len(signals)
    lines += [
        "",
        f"  {n_pass}/{n_total} core claims validated",
        "  → Recommended next step: " + (
            "proceed to adaptive attack testing (Test 5)"
            if n_pass >= 2 else
            "revisit layer selection and projection strategy before proceeding"
        ),
        "=" * 65, "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run HPS Sentinel validation tests")
    parser.add_argument(
        "--tests", type=str, default="3,4,5",
        help="Comma-separated test numbers to run (e.g. --tests 3,4,5)"
    )
    args = parser.parse_args()
    selected = set(args.tests.split(","))

    base = os.path.dirname(os.path.abspath(__file__))
    test_modules = {
        "1": ("test1_gromov_delta",    os.path.join(base, "test1_gromov_delta.py")),
        "2": ("test2_hierarchy",       os.path.join(base, "test2_hierarchy.py")),
        "3": ("test3_traced_space",    os.path.join(base, "test3_traced_space.py")),
        "4": ("test4_baseline_comparison", os.path.join(base, "test4_baseline_comparison.py")),
        "5": ("test5_hps_full",        os.path.join(base, "test5_hps_full.py")),
    }

    print(f"\n{'═'*60}")
    print(f"  HPS SENTINEL — Validation Suite")
    print(f"  Model:  {config.MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Tests:  {', '.join(sorted(selected))}")
    print(f"{'═'*60}\n")

    test_results = {}
    total_t0 = time.time()

    for num in sorted(selected):
        if num not in test_modules:
            print(f"[run_all] Unknown test number: {num}, skipping.")
            continue
        name, path = test_modules[num]
        print(f"\n{'─'*60}")
        print(f"  Running Test {num}: {name}")
        print(f"{'─'*60}")
        test_results[num] = run_test(name, path, num)
        status = test_results[num]["status"]
        elapsed = test_results[num]["elapsed_s"]
        print(f"\n  Test {num} finished: {status} in {elapsed}s")

    total_elapsed = round(time.time() - total_t0, 1)
    print(f"\n[run_all] All tests completed in {total_elapsed}s")

    # ── Summary ──
    summary_txt = generate_summary(test_results)
    print(summary_txt)

    # Save machine-readable summary
    summary_json = {
        "config":       {"model": config.MODEL_NAME, "device": config.DEVICE},
        "test_results": test_results,
        "total_elapsed_s": total_elapsed,
    }
    path_json = os.path.join(config.RESULTS_DIR, "summary_report.json")
    with open(path_json, "w") as f:
        json.dump(summary_json, f, indent=2)

    # Save human-readable summary
    path_txt = os.path.join(config.RESULTS_DIR, "summary_report.txt")
    with open(path_txt, "w") as f:
        f.write(summary_txt)

    print(f"[run_all] Reports saved:")
    print(f"  {path_json}")
    print(f"  {path_txt}")


if __name__ == "__main__":
    main()
