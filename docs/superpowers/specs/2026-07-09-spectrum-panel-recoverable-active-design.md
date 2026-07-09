# Recoverable Spectrum Panel + Show/Active Selection — Design Spec

Date: 2026-07-09

## Purpose

Two problems with the "Loaded Spectra" dock panel (added in the
multi-spectrum feature): closing it via its own `[x]` button leaves no way
to bring it back, and there's currently only one per-spectrum state
("visible", a plain checkbox). This adds a second, independent state:
exactly one spectrum can be marked "active" — the one future peak-finding
and curve-fitting operations (the eventual point of the
"PeakFinderFitting" project) will act on. No fitting logic itself is
implemented here; this only adds the selection mechanism and its data
model.

## Panel recoverability

Add a checkable `View → Loaded Spectra` menu action, wired bidirectionally
to the dock widget's visibility:
- `action.toggled` → `dock.setVisible(...)`
- `dock.visibilityChanged` → `action.setChecked(...)`

This way, closing the dock via its own `[x]` un-checks the menu action
(reflecting reality), and re-checking the menu action always brings the
panel back — there is no longer a dead end.

## Show / Active per spectrum

Each row in the spectrum list becomes a small custom widget (via
`QListWidget.setItemWidget`) instead of a single checkable list item:

- **Show** — a `QCheckBox`, behaves exactly as the current single checkbox
  does (any number of spectra can be shown at once; controls the overlay
  plot).
- **Active** — a `QRadioButton`. All rows' radio buttons share one
  `QButtonGroup`, so selecting one automatically deselects any other —
  Qt enforces "only one at a time" without custom bookkeeping.
- Color swatch and filename remain, as today.

`LoadedSpectrum` gains a new `active: bool` field (default `False`).
`MainWindow` has no separate "active spectrum" field — `active` lives on
each `LoadedSpectrum` instance; at most one is ever `True`.

**Defaults and edge cases:**
- The first spectrum loaded (when the list goes from empty to non-empty)
  becomes active automatically.
- Loading additional spectra afterward does not change which one is
  active.
- Removing the currently-active spectrum makes the next remaining
  spectrum (if any) active automatically, so there's always an active
  spectrum whenever at least one is loaded, and never one when the list
  is empty.
- No visual change to the plot itself for "active" — no special line
  styling. The radio button state in the panel is the only indicator, for
  now.

## Out of scope

- Any actual fitting/peak-finding behavior tied to the active spectrum —
  this spec only adds the selection mechanism and data field.
- Persisting which spectrum was active across app restarts.
