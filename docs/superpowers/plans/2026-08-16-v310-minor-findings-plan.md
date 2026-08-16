# v3.1.0 Minor findings — disposition

**Goal:** close out the 24 Minor findings from the v3.0.0 full-codebase audit, the remainder of the user's "fix critical and important findings, and afterwards try to fix the rest" instruction. All 5 Critical and all 18 Important findings were completed earlier (see `2026-08-13-v310-critical-important-fixes-plan.md`, Tasks 1–23).

**Source of the findings:** the audit artifact (https://claude.ai/code/artifact/ab9ece47-327d-418e-8bee-34f76778622d). No local copy existed — it was re-fetched and the Minor section extracted verbatim. Every finding was then re-verified against current code before acting, because the audit's line numbers predate 23 tasks of fixes.

**Outcome: 3 were already fixed, 16 fixed here, 3 deliberately not changed (with reasons), 2 blocked.**

## Already fixed by the Critical/Important cycle — verified, no action

| # | Finding | Evidence |
|---|---------|----------|
| M9 | Negative N42 run-lengths silently accepted | `n42_io.py` now guards `0 <= run_length <= _MAX_RUN_LENGTH`; the Critical-3 fix covered the negative case too |
| M13 | Matrix canvas styling stale on theme toggle | `matrix_heatmap._refresh_theme` / `matrix_panel._apply_theme` exist and are driven by `main_window._apply_theme`'s loop (Important 13 / Task 18) |
| M14 | Two comments naming the renamed `_plot_projection` | Zero occurrences remain in `matrix_heatmap.py` |

## Fixed

| # | Finding | What was done |
|---|---------|---------------|
| M10 | Calibration coefficients accept `nan`/`inf` | Validated in `Calibration.__post_init__` — the one choke point every construction path funnels through — plus an earlier check in `read_coefficients_file` so the message can name the file. **The only Minor finding with user-visible wrong behaviour.** Proven: 17 new tests fail without it |
| M15 | Active-spectrum lookup duplicated | 16 sites (8 `main_window.py`, 8 `fit_mode.py`, +1 aliased site found later) → `spectrum.active_spectrum()`. Module-level, not a `MainWindow` method: `MatrixPanel` duck-types the same surface for `FitModeController`, so a method would silently join that contract |
| M17 | `QButtonGroup` leaked per list rebuild | `setParent(None)` before rebinding; deterministic where `deleteLater()` needs an event-loop turn the rebuild path may not reach. The old comment claimed the opposite of the measured behaviour. Test counts live `QObject` children |
| M18 | Premature redraw on N42 load | Embedded calibration is now returned from `_try_load_spectrum` and applied by `_load_files` once the list is complete. Verified 2 redraws → 1 |
| M5 | Artist-removal loop duplicated | → `_remove_progress_artists()` |
| M6 | `update_results_list()` row population duplicated | → `_populate_results_row()`; both branches were byte-identical |
| M4 | Both value formatters duplicate the conversion | → `_to_energy()`. Kept because the duplicated part is the *error-propagation math*, not the formatting; the two output shapes the spec requires are unchanged |
| M8 | Background preview recomputes per mark click | `compute_background` is region-bounded, so the only size-scaling cost was the full-length `np.arange`. `peak_fit.channel_indices(n)` caches it **by length** — safe because the channel axis never depends on counts, so in-place mutation (Multiply/Add/Subtract) cannot stale it. Returned read-only |
| M12 | Discarded redraw on X/Y projection switch | `CutController.clear(redraw=False)` for callers that replot immediately. Clear Marks keeps its draw. Verified 2 draws → 1 |
| M7 | `draw_committed_fits()` is the file's largest method | Split into `_fit_draw_context()` + `_draw_result_regions()` / `_draw_integration_annotation()` / `_draw_fit_curves()`, leaving a 20-line loop that names the three concerns. Call order is preserved exactly, because matplotlib layers artists in call order. **Proven identical by artist-level comparison**, not just by tests — see below |
| M19 | Private matplotlib API + unbounded pin | `matplotlib>=3.8,<3.12`, with the reason recorded at the pin |
| M20 | Spec text says Ctrl+C deletes | Addendum recording that it was implemented then deliberately reverted to hide-not-delete |
| M21 | Figure 7 caption says "gated rows" | → "gated columns"; `compute_cut` slices a column range for `axis='x'` |
| M22 | No Python floor declared | Recorded in `pyproject.toml`. Deliberately a comment, not a PEP 621 `[project]` table — that would require a `version` field, creating a second source of truth able to drift from `installer.iss` |
| M23 | Dead `*.AppDir/` ignore entry | Removed |
| M24 | `.gitattributes` comment cites appimagetool | Re-worded to name dpkg-deb/rpmbuild; the LF rule itself is still required |

## Deliberately not changed

- **M3 — model function rebuilds dict/f-string state per evaluation.** Implemented, verified bit-identical across 96 configurations, then **measured: 1.02× (≈1.5%)** on an 8-peak/512-point model. The cost is dominated by the numpy `exp`/`erfc` work, matching the audit's own "secondary, not a primary driver" note. Reverted — ~30 lines of extra indirection in numerically-critical code is not worth 1.5%.
- **M11 — matrix panel redraws twice on calibration change.** The second redraw is deliberate and documented at `matrix_panel.py`'s `_apply_calibration_change` as belt-and-suspenders, explicitly "not reliant on the `_matrix_panels` registration loop for correctness". Removing it to save ~30 ms (invisible; Qt coalesces the paint) would make correctness depend on that registration.
- **M16 — spectrum-list panel rebuilt wholesale per add/remove.** O(M²) only across M *sequential single-spectrum* operations; batch open already rebuilds once. At the audit's own "dozens of spectra" scale this is microseconds. An incremental rewrite of a core UI path is exactly the shape of change that has produced stale-widget bugs in this project before. The unbounded part of this finding was the `QButtonGroup` leak, which **is** fixed (M17).

## Blocked

- **M1 (initial sigma guess ~2× TV's scale) and M2 (Marquardt half-step fallback skips re-applying damping).** Both are TV-parity claims, and this project's standing rule is to verify parity against `tv-1.9.13/`'s real C source rather than any spec's prose. That vendored tree is untracked and did not come across in the machine move, so the claims cannot be checked here. Changing fitting math on the strength of a prose summary is exactly what that rule exists to prevent. **Resume when `tv-1.9.13/` is available**, then re-derive both against `vsCurFit.c`/`VsFitFct.c`.

## Verification

**M7 specifically** was verified beyond the test suite, because no test asserts rendered output and a drawing-order mistake would be invisible to them: a scenario covering all three result kinds (a two-peak Fit, an Integration with background, an Integration without) plus a hidden result was rendered before and after the split, and every artist matplotlib produced was captured — type, xy data, color, linestyle, linewidth, alpha, annotation text and anchor, and z-order — then compared element-by-element in list order. **31 artists, identical in both cases** (20 with a result hidden). That is what makes the split safe to claim as behaviour-preserving; the 363 passing tests in the affected suites do not, on their own, prove it.

Full suite after all changes: **810 passed, 2 xfailed, 1 failed** — up from 782 passed (28 tests added). The single failure is `test_load_mtx_decodes_real_fixture_reasonably_fast`, a machine-calibrated wall-clock assertion unrelated to these changes; see that test's own note and project memory. Each behavioural fix was additionally proven to fail before its fix (M10: 17 failures; M17, M18, M12: asserted counts).
