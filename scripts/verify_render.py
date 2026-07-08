"""One-off dev helper: render a histogram file to a PNG for visual sanity-checking.

Usage: QT_QPA_PLATFORM=offscreen python scripts/verify_render.py <histogram.txt> <output.png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    histogram_path, output_path = sys.argv[1], sys.argv[2]
    app = QApplication([])
    window = MainWindow()
    window._load_file(histogram_path)
    window.figure.savefig(output_path)


if __name__ == "__main__":
    main()
