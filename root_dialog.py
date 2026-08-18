"""Qt dialog for choosing what to load out of a ROOT file.

Every other format this app reads holds exactly one thing, so opening it
is a file dialog and nothing more. A ROOT file holds a directory TREE of
named objects -- the acquisition file this was developed against carries
seven histograms across five directories -- so the user has to be shown
what is in there and asked which they want.
"""

import os

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class RootObjectDialog(QDialog):
    """`result_spectra` and `result_matrix` are set only after a
    successful OK, and stay empty/None if cancelled.

    Spectra are multi-selectable, matrices are not, and the two cannot be
    mixed in one selection -- a matrix opens its own panel while spectra
    join the spectrum list, so combining them in a single OK would mean
    two unrelated things happening at once. Selecting a matrix alongside
    spectra reports that rather than silently dropping one of them.
    """

    def __init__(self, parent, path, objects):
        super().__init__(parent)
        self.setWindowTitle(f"Open ROOT File - {os.path.basename(path)}")
        self.result_spectra = []
        self.result_matrix = None
        self._objects = list(objects)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select one or more 1D histograms to load as spectra, "
            "or a single 2D histogram to open as a matrix."
        ))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Type", "Opens as"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setRootIsDecorated(False)
        for object_path, classname, kind in self._objects:
            # ROOT appends a cycle number (";1") to every key. It is part of
            # the real key and has to be kept for lookup, but showing it
            # would be noise to a user who has never heard of cycles.
            display = object_path.split(";")[0]
            item = QTreeWidgetItem([
                display, classname, "Spectrum" if kind == "spectrum" else "Matrix",
            ])
            item.setData(0, 0x0100, object_path)  # Qt.ItemDataRole.UserRole
            self.tree.addTopLevelItem(item)
        self.tree.resizeColumnToContents(0)
        layout.addWidget(self.tree)

        self.status = QLabel("")
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
            # Focus so Qt draws that selection in its ACTIVE palette. An
            # unfocused list renders a selection in a pale inactive grey,
            # which reads as nothing being selected -- the row genuinely
            # was selected and OK would have loaded it, but there was no
            # way to tell by looking. The stylesheet now also colours the
            # inactive state, so this is belt and braces rather than the
            # only defence.
            self.tree.setFocus()

    def _selected(self):
        chosen = []
        for item in self.tree.selectedItems():
            object_path = item.data(0, 0x0100)
            kind = "matrix" if item.text(2) == "Matrix" else "spectrum"
            chosen.append((object_path, kind))
        return chosen

    def _on_accept(self):
        chosen = self._selected()
        if not chosen:
            self.status.setText("Select at least one object.")
            return

        matrices = [p for p, kind in chosen if kind == "matrix"]
        spectra = [p for p, kind in chosen if kind == "spectrum"]

        if matrices and spectra:
            self.status.setText(
                "Select either spectra or a single matrix, not both."
            )
            return
        if len(matrices) > 1:
            self.status.setText("Only one matrix can be opened at a time.")
            return

        self.result_matrix = matrices[0] if matrices else None
        self.result_spectra = spectra
        self.accept()
