# Close Spectrum — Design Spec

Date: 2026-07-30

## Purpose

User request: add a more discoverable way to remove a loaded spectrum from the program (in-memory only — the file on disk is never touched). A right-click "Remove" context-menu action on the Loaded Spectra panel already provides exactly this, but it's not discoverable without knowing to right-click a row in that panel.

## Current State (audited before designing)

`main_window.py`'s `_on_spectrum_context_menu` (right-click on a row in the "Loaded Spectra" dock) already offers "Remove": given the right-clicked row's spectrum path, it drops that spectrum from `self.spectra`, promotes another loaded spectrum to active if the removed one was active, refreshes the spectrum list UI, and replots. No confirmation dialog. The file on disk is never touched — this only ever mutates the in-memory `self.spectra` list. There is currently no test coverage for this action at all (confirmed by search — no test references `_on_spectrum_context_menu` or exercises the "Remove" path).

## Resolved Ambiguities

These were established through discussion, not assumed:

- **What "more discoverable" means**: add a menu item and a keyboard shortcut, not a redesign of the existing right-click path. The user confirmed they want discoverability only — the existing removal behavior itself is already correct.
- **Which spectrum a menu/shortcut action removes**: the active spectrum — consistent with every other Operations-menu command (Multiply, Rebin, Fit, Integrate, Clear all act on whichever spectrum is currently active via its radio button), rather than introducing a second, list-row-based "selection" concept alongside "active."
- **Menu location**: File menu, between Save Spectrum and Recent Files — this is a spectrum-management action like Open/Save, not a data-transforming command like Multiply/Rebin/Normalize (which live in Operations).
- **Naming**: "Close Spectrum," not "Remove Spectrum" — avoids reading as disk-related to someone skimming the File menu, and matches the common "File > Close [current document]" convention. `Ctrl+W` is the standard shortcut for this action in most applications and is unused anywhere in this codebase.
- **Confirmation**: none — matches this app's already-established philosophy (`Ctrl+C` deletes committed fits with no prompt; the file on disk, untouched by any of this, is the safety net rather than a confirmation dialog).

## Behavior

**Trigger**: `Ctrl+W`, or File > Close Spectrum. Disabled (matching Multiply/Rebin/Save Spectrum's existing pattern) when there's no active spectrum.

**Refactor**: the existing right-click Remove's logic (currently inline in `_on_spectrum_context_menu`) is extracted into a new shared method, `_remove_spectrum(self, path)`, taking the path of the spectrum to remove: computes whether the removed spectrum was active, filters it out of `self.spectra`, promotes `self.spectra[0]` to active if the removed spectrum was active and any spectra remain, refreshes the spectrum list, and replots. `_on_spectrum_context_menu` is updated to call this with the right-clicked row's path instead of inlining the same four lines — its own observable behavior is unchanged, this is a pure extraction.

**New method**: `_close_active_spectrum(self)` finds the currently active spectrum (`next((s for s in self.spectra if s.active), None)`); if none, no-ops; otherwise calls `_remove_spectrum(active.path)`.

**New menu item**: "Close Spectrum" (`QAction`, shortcut `Ctrl+W`, starts disabled), added to the File menu between the existing Save Spectrum action and the Recent Files submenu, wired to `_close_active_spectrum`. `_update_operations_availability` gains `self.close_spectrum_action.setEnabled(active is not None)`, matching Multiply/Rebin/Save Spectrum's existing pattern exactly.

## Testing

- `_remove_spectrum(path)`, tested directly: removes the spectrum matching `path` and no other; when the removed spectrum was active, another loaded spectrum (if any remain) becomes active; when the removed spectrum was not active, the previously-active spectrum's `.active` state is untouched. This closes the pre-existing coverage gap on the right-click path (which calls the same method) as a side effect of the refactor.
- `_close_active_spectrum`: removes the active spectrum from `self.spectra`; promotes another loaded spectrum to active afterward when one remains; does not raise when zero spectra are loaded.
- `close_spectrum_action`: disabled with no active spectrum, enabled once one exists, correct shortcut (`Ctrl+W`), present in the File menu at the correct position (between Save Spectrum and Recent Files).

## Out of Scope

- Any change to the existing right-click "Remove" context-menu action's own observable behavior or wording — stays exactly as-is; only its underlying logic moves into a shared, reusable method.
- Closing/removing multiple spectra at once.
- A "close all spectra" command.
- Any confirmation dialog before closing.
