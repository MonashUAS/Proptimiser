'''
Proptimiser graphical interface.

    python app.py                   # start from the default propeller
    python app.py currawong_v2      # open designs/currawong_v2.yaml and propellers/currawong_v2.state.json

'''


import os
import signal
import sys

# pyvistaqt (through qtpy) must use the same Qt binding as matplotlib
os.environ.setdefault("QT_API", "pyside6")

# Internal Imports: init applies the JAX configuration before anything else imports JAX
import init

import matplotlib
matplotlib.use("QtAgg")
# Figures are embedded in tabs rather than full windows, so use a smaller type scale
matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.session import Session


def main(argv):
    app = QApplication(argv)
    app.setApplicationName("Proptimiser")

    if len(argv) > 1:
        prop = init.Propeller.load(argv[1])
        path = init.design_paths(argv[1])[0]
    else:
        path, prop = None, init.Propeller.default()

    window = MainWindow(Session(prop, path))

    # proptimiser installs a "stop after this iteration" Ctrl+C handler for the CLI. In the GUI the Stop
    # button does that, and Ctrl+C in the terminal closes the application.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    screen = app.primaryScreen().availableGeometry()
    window.resize(int(0.9 * screen.width()), int(0.9 * screen.height()))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
