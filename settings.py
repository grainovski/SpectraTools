from PySide6.QtCore import QSettings

ORG_NAME = "PeakFinderFitting"
APP_NAME = "HistogramViewer"
MAX_RECENT_FILES = 8


class Settings:
    def __init__(self):
        self._settings = QSettings(ORG_NAME, APP_NAME)

    def last_folder(self) -> str:
        value = self._settings.value("last_folder", "")
        return value if isinstance(value, str) else ""

    def set_last_folder(self, folder: str) -> None:
        self._settings.setValue("last_folder", folder)
        self._settings.sync()

    def recent_files(self) -> list:
        value = self._settings.value("recent_files", [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def add_recent_file(self, path: str) -> None:
        files = self.recent_files()
        if path in files:
            files.remove(path)
        files.insert(0, path)
        self._settings.setValue("recent_files", files[:MAX_RECENT_FILES])
        self._settings.sync()

    def remove_recent_file(self, path: str) -> None:
        files = self.recent_files()
        if path in files:
            files.remove(path)
            self._settings.setValue("recent_files", files)
            self._settings.sync()
