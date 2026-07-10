def test_qapp_fixture_provides_a_running_application(qapp):
    from PySide6.QtWidgets import QApplication
    assert isinstance(qapp, QApplication)
