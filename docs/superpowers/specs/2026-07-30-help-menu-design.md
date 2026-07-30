# Help Menu — Design Spec

Date: 2026-07-30

## Purpose

User request: add a `Help` menu, rightmost on the menu bar, with three sections — **HowTo**, **Knowledge Database**, and **About** — each opening a page in the system's default web browser. HowTo documents every keyboard shortcut (grouped by menu) plus short step-by-step manuals for every operation the app supports. Knowledge Database explains, with graphical examples, how fits are actually performed and what each fit parameter means. About shows program name, version, build date, and copyright to Georgi Rainovski.

**Reference example**: the user pointed to the TV nuclear-spectroscopy tool's user manual (`apps.ikp.uni-koeln.de/~fitz/viewspectra/Tv_user-manual/`) as a "for example." That manual is a large, multi-page, LaTeX-generated hierarchical reference (command-by-command, with a separate hotkey-table page) — appropriate for TV's much larger command surface. This app has ~20 shortcuts and a dozen operations total, so HowTo is designed as a single scrollable page per section (shortcut tables + task manuals together) rather than TV's multi-page structure — same spirit (complete shortcut coverage grouped sensibly, step-by-step operation guides), sized to this app.

## Current State (audited before designing)

No `Help` menu exists. The menu bar is `File`, `View`, `Operations` (`main_window.py::_build_menu`). No About dialog, no bundled documentation, no version-string constant anywhere in the codebase. `packaging/windows/build.ps1` runs PyInstaller with no `--add-data` — the project has never bundled non-Python data files; the exe is Python source + dependencies only.

The full current shortcut inventory (verified by grep across `main_window.py` and `fit_mode.py`, not from memory):

| Menu/area | Shortcut | Action |
|---|---|---|
| File | Ctrl+O | Open |
| File | Ctrl+S | Save Spectrum... |
| File | Ctrl+W | Close Spectrum |
| File | Ctrl+Q | Exit |
| View | Ctrl+G | Log scale Y (toggle) |
| View | Ctrl+1 | Toggle Spectra panel |
| View | Ctrl+D | Dark theme (toggle) |
| View | Ctrl+2 | Toggle Fit Results panel |
| View | Ctrl+3 | Toggle Fit Parameters panel |
| Operations | Ctrl+L | Calibration... |
| Operations | Ctrl+T | Toggle Calibration Active |
| Operations | Ctrl+M | Multiply by Factor... |
| Operations | Ctrl+R | Rebin by Factor... |
| Operations | Ctrl+N | Normalize Spectra |
| Plot toolbar | Ctrl+= | Zoom In X |
| Plot toolbar | Ctrl+- | Zoom Out X |
| Plot toolbar | Ctrl+0 | Show Full Spectrum |
| Fitting | Ctrl+F | Fit |
| Fitting | Ctrl+C | Clear (deletes active spectrum's fits) |
| Fitting | Ctrl+Shift+C | Clear All Fits |
| Fitting | Ctrl+E | Export All Fits |
| Fitting | Ctrl+I | Integrate |
| Fitting | B / R / P (bare keys, click-to-mark) | Mark background region / fit region / peak position |

Supported spectrum file formats (from the Open/Save dialog filters): `.txt`, `.spe`, `.spk`.

Fit model (from `peak_fit.py`, ported from `srcRW/gf3_subs.c`'s Hypermet left tail): a linear background (slope + intercept) plus, per peak, a Gaussian core blended with an exponential low-energy tail: `(1-r)*exp(-w^2) + r*exp(dx/beta)*erfc(w+y)/erfc(y)`. Peaks in a multiplet share one FWHM. Parameters: `position`, `sigma`/FWHM (shared), `amplitude`, `tail_fraction` (r), `tail_beta` (β), plus the background's slope/intercept. `FitResult`/`PeakResult` carry both gross and net area with uncertainties. Calibration (`calibration.py`) is linear or quadratic, channel→keV, applied app-wide.

## Resolved Ambiguities

Established through discussion, not assumed:

- **Graphical examples**: generated originally, by this app's own matplotlib stack, using its real fit math — not stock/external images, not text-only.
- **External links**: none. The Knowledge Database is fully self-contained despite the original request allowing external links — the user opted out when asked.
- **Build date**: auto-stamped by `build.ps1` at packaging time (not manually maintained, not tied to git tag date).
- **Architecture**: pages are generated at runtime as self-contained HTML strings (figures embedded as base64 `data:` URIs) and opened via `QDesktopServices.openUrl` — not static files bundled through PyInstaller `--add-data`. This introduces no new PyInstaller bundling surface: matplotlib is already a hard dependency, so figure generation needs nothing new to ship.
- **Menu labels**: `HowTo`, `Knowledge Database`, `About` (Title Case, matching this app's existing menu-label convention).
- **Shortcuts**: `F1` on HowTo (the conventional "open help" key, and this app has audited every other command for a shortcut this session — leaving Help unbound would be the outlier). No shortcut on Knowledge Database or About — reference material, not a hurried action.

## Architecture

**New module `help_content.py`**: builds each page's complete HTML as a Python string.
- `build_howto_html() -> str`
- `build_knowledge_database_html() -> str` (calls into `help_figures.py` for each embedded figure)
- `build_about_html() -> str` (imports `build_info` for `VERSION`/`BUILD_DATE`, falling back to `VERSION = "dev"`, `BUILD_DATE = "development build"` on `ImportError` — never raises when `build_info.py` doesn't exist, e.g. running from source without having built)
- `open_help_page(html: str) -> None`: writes `html` to a fresh `tempfile.NamedTemporaryFile(suffix=".html", delete=False)` and opens it via `QDesktopServices.openUrl(QUrl.fromLocalFile(path))`. Regenerated fresh every time a Help action fires — no caching, no stale content, negligible cost (a few KB of HTML plus sub-second matplotlib renders).

**New module `help_figures.py`**: one function per figure, each returning PNG bytes (rendered via matplotlib's `Agg` backend into an in-memory `BytesIO`, no file I/O):
- `anatomy_of_a_fit_figure() -> bytes` — a real spectrum with both background regions, the fit region, a peak mark, and the fitted curve shown together; position/FWHM/amplitude/background line annotated directly on the plot.
- `tail_effect_figure() -> bytes` — the same peak shape with `tail_fraction=0` (pure Gaussian) vs. `tail_fraction>0` overlaid, making the low-energy tail visually obvious.
- `multiplet_figure() -> bytes` — 2-3 overlapping peaks fit together with one shared FWHM.
- `calibration_curve_figure() -> bytes` — channel→keV, linear vs. quadratic fit through the same points, residuals shown.

`help_content.py` base64-encodes each figure's bytes and inlines them as `<img src="data:image/png;base64,...">` — the generated HTML file has no external file dependencies at all, so it opens correctly from a temp path regardless of packaged vs. source execution.

**`main_window.py`** (`_build_menu`): new `Help` menu, added after `Operations`, with three `QAction`s wired to three new thin handler methods (`_open_howto`, `_open_knowledge_database`, `_open_about`), each calling the matching `help_content.py` builder then `open_help_page`.

**`build_info.py`**: generated by `build.ps1`, gitignored, never committed — same convention already established for `dist/`, `build/`, `packaging/*/output/`, `releases/`. `build.ps1` reads `AppVersion` out of `installer.iss` via `Select-String`, takes today's date (`Get-Date -Format "yyyy-MM-dd"`), and writes:
```python
VERSION = "1.0.0"
BUILD_DATE = "2026-07-30"
```
to `build_info.py` in the repo root immediately before invoking PyInstaller, so the exact stamped values get compiled into that build's exe.

## Content — HowTo Page

**Part 1, shortcut reference**: the full table from "Current State" above, rendered as HTML tables grouped by the same five areas (File, View, Operations, Plot toolbar, Fitting) — not one giant flat table.

**Part 2, short task manuals**, one subsection each, step-by-step and concrete (not just "see the menu"):
1. Loading a spectrum — Open, the three supported formats, Recent Files
2. Working with multiple spectra — the "active" spectrum concept (radio-button selection), switching active, right-click Remove vs. Close Spectrum's active-only scope
3. Saving a spectrum — the format-choice dialog
4. Calibrating the energy axis — linear vs. quadratic, loading a calibration vs. toggling it active, what changes on the plot and in the Fit Results panel
5. Multiply by Factor / Rebin by Factor / Normalize Spectra — one paragraph each
6. **Performing a fit** (the core workflow): press `B` twice for the two background regions, `R` for the fit region, `P` per peak, then `Ctrl+F`; reading the Fit Results/Fit Parameters panels; `Ctrl+C` to clear the active spectrum's fits, `Ctrl+Shift+C`/`Ctrl+E` for all spectra
7. Integration — `Ctrl+I`, and how it differs from fitting
8. View options — log scale, dark theme, panel toggles, zoom, full spectrum

## Content — Knowledge Database Page

Written sections, each cross-referencing one of the four figures from `help_figures.py`:
1. Why gamma-spectroscopy peak fits use a tailed Gaussian (brief, physical motivation — not a textbook chapter)
2. The Hypermet shape function this app actually uses (the formula from `peak_fit.py`, in words and in the same notation as the source), illustrated by the anatomy-of-a-fit figure
3. Parameter meaning table: position, FWHM (shared across a multiplet), amplitude, tail fraction `r`, tail β, background slope/intercept — cross-referenced to the tail-effect and multiplet figures
4. Gross area vs. net area — what the Fit Results panel's two area columns mean, and how their uncertainties are derived
5. Integration vs. fitting — what `Ctrl+I` computes and when it's the right tool instead of a full fit
6. Calibration math — linear/quadratic, and how it changes displayed units (channels → keV) for position and FWHM, illustrated by the calibration-curve figure

No outbound links anywhere on this page.

## Content — About Page

Program name (SpectraTools), version, build date, and `Copyright © Georgi Rainovski` — short, centered, no logo/icon (out of scope — nothing in the request calls for one, and designing one is a separate task).

## Testing

- `help_content.py`: each `build_*_html()` function's output contains every expected shortcut string, every parameter name from the table above, "Georgi Rainovski", and the version/date values (mocking `build_info` presence/absence for the About test, including the no-`build_info.py` fallback path).
- `help_figures.py`: each figure function returns non-empty, valid PNG bytes (`PIL.Image.open` round-trip or a magic-bytes check) — a smoke check, not pixel comparison.
- `open_help_page`: monkeypatch `QDesktopServices.openUrl` (same monkeypatching pattern already used for `QMessageBox.warning` elsewhere in this codebase) and assert it's called with a `QUrl` pointing at a real, non-empty temp file — no test ever spawns an actual browser.
- Menu: `Help` is the last menu on the bar; contains exactly `HowTo`, `Knowledge Database`, `About` in that order; `HowTo` has shortcut `F1`; the other two have none.
- `build.ps1` changes are not unit-testable (PowerShell, runs only at build time) — covered by the existing manual smoke-test-the-built-exe step already used for every installer build in this project.

## Out of Scope

- An About dialog/logo/icon.
- Any external links or fetched content — everything is generated locally and self-contained.
- Static bundled HTML/image files or any PyInstaller `--add-data` usage.
- In-app help (tooltips, `?` buttons, contextual hints) — this is only the three Help-menu pages.
- Localization/translation of help content.
- A search feature within the help pages.
- Keeping previously-generated temp HTML files around — each open regenerates fresh and the OS temp directory is responsible for eventual cleanup, same as any other throwaway temp file.
