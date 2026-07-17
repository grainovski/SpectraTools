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
    assert names[save_index + 1:] == ["Zoom In X", "Zoom Out X", "Show Full Spectrum"]
