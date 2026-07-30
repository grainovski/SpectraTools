from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMenu

from main_window import MainWindow


def _menu_named(main_window, title):
    # Deliberately not `next(a.menu() for a in main_window.menuBar().actions()
    # if a.text() == title)`: under PySide6 6.8.3, a QMenu fetched via
    # QAction.menu() has its underlying C++ object torn down for real once
    # every Python reference to the specific QAction wrapper it came from
    # is dropped, regardless of the QMenu's real Qt parent (the menu bar)
    # still being alive. findChildren sidesteps that lifetime issue
    # entirely. (Same helper as tests/test_operations_menu.py.)
    for menu in main_window.menuBar().findChildren(QMenu):
        if menu.title() == title:
            return menu
    raise AssertionError(f"no menu titled {title!r}")


def test_help_menu_is_last_on_the_menu_bar(qapp):
    main_window = MainWindow()
    menu_bar_actions = main_window.menuBar().actions()
    assert menu_bar_actions[-1].text() == "&Help"


def test_help_menu_has_the_three_actions_in_order(qapp):
    main_window = MainWindow()
    help_menu = _menu_named(main_window, "&Help")
    texts = [a.text() for a in help_menu.actions()]
    assert texts == ["HowTo", "Knowledge Database", "About"]


def test_howto_action_has_f1_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.howto_action.shortcut() == QKeySequence("F1")


def test_knowledge_database_and_about_actions_have_no_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.knowledge_database_action.shortcut().isEmpty()
    assert main_window.about_action.shortcut().isEmpty()


def _recording_open_help_page(calls):
    # Returns True (matching open_help_page's real success contract) --
    # a mock returning None/falsy would spuriously also exercise the
    # failure-status-message path these tests aren't testing.
    def _open(html):
        calls.append(html)
        return True

    return _open


def test_howto_action_opens_the_howto_page(qapp, monkeypatch):
    import main_window as main_window_module
    calls = []
    monkeypatch.setattr(main_window_module, "open_help_page", _recording_open_help_page(calls))
    window = MainWindow()
    window.howto_action.trigger()
    assert len(calls) == 1
    assert "HowTo" in calls[0]


def test_knowledge_database_action_opens_the_knowledge_database_page(qapp, monkeypatch):
    import main_window as main_window_module
    calls = []
    monkeypatch.setattr(main_window_module, "open_help_page", _recording_open_help_page(calls))
    window = MainWindow()
    window.knowledge_database_action.trigger()
    assert len(calls) == 1
    assert "Knowledge Database" in calls[0]


def test_about_action_opens_the_about_page(qapp, monkeypatch):
    import main_window as main_window_module
    calls = []
    monkeypatch.setattr(main_window_module, "open_help_page", _recording_open_help_page(calls))
    window = MainWindow()
    window.about_action.trigger()
    assert len(calls) == 1
    assert "About" in calls[0]


def test_howto_action_shows_status_message_when_open_help_page_fails(qapp, monkeypatch):
    import main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_help_page", lambda html: False)
    window = MainWindow()
    messages = []
    monkeypatch.setattr(
        window.fit_controller, "_show_status_message",
        lambda message, duration_ms: messages.append(message),
    )
    window.howto_action.trigger()
    assert len(messages) == 1
    assert "HowTo" in messages[0]


def test_knowledge_database_action_shows_status_message_when_open_help_page_fails(qapp, monkeypatch):
    import main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_help_page", lambda html: False)
    window = MainWindow()
    messages = []
    monkeypatch.setattr(
        window.fit_controller, "_show_status_message",
        lambda message, duration_ms: messages.append(message),
    )
    window.knowledge_database_action.trigger()
    assert len(messages) == 1
    assert "Knowledge Database" in messages[0]


def test_about_action_shows_status_message_when_open_help_page_fails(qapp, monkeypatch):
    import main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_help_page", lambda html: False)
    window = MainWindow()
    messages = []
    monkeypatch.setattr(
        window.fit_controller, "_show_status_message",
        lambda message, duration_ms: messages.append(message),
    )
    window.about_action.trigger()
    assert len(messages) == 1
    assert "About" in messages[0]
