# Close Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a discoverable File-menu item + `Ctrl+W` shortcut that closes (removes from the program, never touches the file on disk) the currently active spectrum — the same in-memory-only removal the existing right-click "Remove" already provides, made reachable without needing to right-click a specific row.

**Architecture:** Extract the existing right-click Remove's inline logic (in `main_window.py`'s `_on_spectrum_context_menu`) into a shared `_remove_spectrum(path)` method. A new `_close_active_spectrum()` method finds the active spectrum and calls the same shared method — mirroring exactly how "Clear All Fits" was already split into a shared method reused by both its context-menu action and its persistent shortcut, earlier in this project's history.

**Tech Stack:** PySide6 (`QAction`), same as the rest of `main_window.py`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-30-close-spectrum-design.md` — read this first for full rationale; this plan implements it and does not repeat its reasoning.

---

### Task 1: Extract `_remove_spectrum`, add `_close_active_spectrum`, wire up the File menu item

**Files:**
- Modify: `main_window.py` (`_on_spectrum_context_menu`, `_build_menu`, `_update_operations_availability`, plus two new methods)
- Test: `tests/test_operations_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_operations_menu.py`:

```python
def test_remove_spectrum_removes_only_the_matching_path(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = False
    spectrum_b.active = True

    main_window._remove_spectrum(spectrum_a.path)

    assert main_window.spectra == [spectrum_b]
    assert spectrum_b.active is True


def test_remove_spectrum_promotes_another_to_active_when_the_active_one_is_removed(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = True
    spectrum_b.active = False

    main_window._remove_spectrum(spectrum_a.path)

    assert main_window.spectra == [spectrum_b]
    assert spectrum_b.active is True


def test_remove_spectrum_leaves_the_program_empty_when_it_was_the_only_one(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._remove_spectrum(spectrum.path)

    assert main_window.spectra == []


def test_close_active_spectrum_removes_the_active_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = False
    spectrum_b.active = True

    main_window._close_active_spectrum()

    assert main_window.spectra == [spectrum_a]
    assert spectrum_a.active is True


def test_close_active_spectrum_does_nothing_with_no_spectra_loaded(qapp):
    main_window = MainWindow()
    main_window._close_active_spectrum()  # must not raise
    assert main_window.spectra == []


def test_close_spectrum_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.close_spectrum_action.isEnabled() is False


def test_close_spectrum_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.close_spectrum_action.isEnabled() is True


def test_close_spectrum_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.close_spectrum_action.shortcut() == QKeySequence("Ctrl+W")


def test_close_spectrum_action_in_file_menu_between_save_and_recent_files(qapp):
    main_window = MainWindow()
    actions = main_window.file_menu.actions()
    assert main_window.close_spectrum_action in actions
    save_index = actions.index(main_window.save_spectrum_action)
    close_index = actions.index(main_window.close_spectrum_action)
    recent_index = actions.index(main_window.recent_menu.menuAction())
    assert save_index < close_index < recent_index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -k "remove_spectrum or close_active_spectrum or close_spectrum_action" -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute '_remove_spectrum'` (and similar for the other new names)

- [ ] **Step 3: Extract `_remove_spectrum` and add `_close_active_spectrum`**

In `main_window.py`, find `_on_spectrum_context_menu` (search by name — line numbers have shifted since earlier tasks touched this file). It currently reads:

```python
    def _on_spectrum_context_menu(self, position):
        item = self.spectrum_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(self.spectrum_list.viewport().mapToGlobal(position))
        if chosen == remove_action:
            path = item.data(Qt.ItemDataRole.UserRole)
            removed_was_active = any(s.path == path and s.active for s in self.spectra)
            self.spectra = [s for s in self.spectra if s.path != path]
            if removed_was_active and self.spectra:
                self.spectra[0].active = True
            self._update_spectrum_list()
            self._plot_data()
```

Replace it with:

```python
    def _on_spectrum_context_menu(self, position):
        item = self.spectrum_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(self.spectrum_list.viewport().mapToGlobal(position))
        if chosen == remove_action:
            path = item.data(Qt.ItemDataRole.UserRole)
            self._remove_spectrum(path)

    def _remove_spectrum(self, path):
        removed_was_active = any(s.path == path and s.active for s in self.spectra)
        self.spectra = [s for s in self.spectra if s.path != path]
        if removed_was_active and self.spectra:
            self.spectra[0].active = True
        self._update_spectrum_list()
        self._plot_data()

    def _close_active_spectrum(self):
        active = next((s for s in self.spectra if s.active), None)
        if active is None:
            return
        self._remove_spectrum(active.path)
```

- [ ] **Step 4: Add the menu item**

In `_build_menu`, find the `save_spectrum_action` block, which currently ends right before `self.recent_menu = self.file_menu.addMenu("Recent Files")`:

```python
        self.save_spectrum_action = QAction("Save Spectrum...", self)
        self.save_spectrum_action.setShortcut("Ctrl+S")
        self.save_spectrum_action.setEnabled(False)
        self.save_spectrum_action.triggered.connect(self._open_save_spectrum_dialog)
        self.file_menu.addAction(self.save_spectrum_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")
```

Change it to:

```python
        self.save_spectrum_action = QAction("Save Spectrum...", self)
        self.save_spectrum_action.setShortcut("Ctrl+S")
        self.save_spectrum_action.setEnabled(False)
        self.save_spectrum_action.triggered.connect(self._open_save_spectrum_dialog)
        self.file_menu.addAction(self.save_spectrum_action)

        self.close_spectrum_action = QAction("Close Spectrum", self)
        self.close_spectrum_action.setShortcut("Ctrl+W")
        self.close_spectrum_action.setEnabled(False)
        self.close_spectrum_action.triggered.connect(self._close_active_spectrum)
        self.file_menu.addAction(self.close_spectrum_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")
```

- [ ] **Step 5: Wire up availability**

In `_update_operations_availability`, change:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

to:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        self.close_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS, all tests

Then run the full suite:

Run: `python -m pytest tests/ -v`
Expected: PASS, every test in the repository (one pre-existing, unrelated, environment-dependent failure — `tests/test_theme.py::test_zoom_icons_and_builtin_save_icon_match_color_in_both_themes` — is expected and not caused by this change)

- [ ] **Step 7: Commit**

```bash
git add main_window.py tests/test_operations_menu.py
git commit -m "feat: add Close Spectrum (Ctrl+W) to remove the active spectrum from the program"
```

---

## Final check

After Task 1, a quick manual smoke test before considering this done:

1. Launch the app, load two spectra.
2. Confirm File > Close Spectrum is present, between Save Spectrum and Recent Files, with `Ctrl+W` shown.
3. Press `Ctrl+W` — confirm the active spectrum disappears from the Loaded Spectra list and the other one becomes active, and the file on disk is untouched (still there, reloadable via Open).
4. Confirm the existing right-click "Remove" on a spectrum row still works exactly as before.
5. Close the last remaining spectrum — confirm the app doesn't crash and the plot goes blank.
