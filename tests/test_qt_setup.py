def test_qapp_fixture_provides_a_running_application(qapp):
    from PySide6.QtWidgets import QApplication
    assert isinstance(qapp, QApplication)


def test_nav_toolbar_drops_unwanted_matplotlib_buttons(qapp):
    from main_window import MainWindow

    main_window = MainWindow()
    names = [a.text() for a in main_window.nav_toolbar.actions() if a.text()]
    assert "Back" not in names
    assert "Forward" not in names
    assert "Zoom" not in names
    assert "Subplots" not in names
    assert "Customize" not in names
    assert "Home" in names
    assert "Pan" in names
    assert "Save" in names


def test_nav_toolbar_places_custom_zoom_actions_right_after_save(qapp):
    from main_window import MainWindow

    main_window = MainWindow()
    names = [a.text() for a in main_window.nav_toolbar.actions() if a.text()]
    save_index = names.index("Save")
    assert names[save_index + 1:] == [
        "Zoom In X", "Zoom Out X", "Show Full Spectrum",
        "Load Calibration...", "Calibration Active",
    ]


def test_nav_toolbar_has_no_stretchy_coordinate_label_between_save_and_zoom(qapp):
    """Regression guard: matplotlib's built-in "coordinates" mouse-
    position label is an expanding widget action with no text -- a
    text-only check (like the test above) doesn't see it, but its
    Expanding size policy visually pushes everything added after it
    (this app's own zoom actions) to hug the toolbar's far-right edge
    instead of sitting right next to Save. This app already shows the
    mouse position in its own status bar, so the label is disabled
    outright (coordinates=False) rather than worked around."""
    from main_window import MainWindow

    main_window = MainWindow()
    assert not hasattr(main_window.nav_toolbar, "locLabel")

    all_actions = main_window.nav_toolbar.actions()
    save_index = next(i for i, a in enumerate(all_actions) if a.text() == "Save")
    # Every action between Save and "Zoom In X" must be a plain
    # separator -- no anonymous (empty-text, non-separator) widget
    # action sitting in between.
    zoom_index = next(i for i, a in enumerate(all_actions) if a.text() == "Zoom In X")
    for action in all_actions[save_index + 1:zoom_index]:
        assert action.isSeparator()
