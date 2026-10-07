## Main window: Setup | Design | Optimise tabs, file menu and a shared log

# External Imports
import sys
import warnings
from pathlib import Path

import yaml

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFontDatabase, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QTabWidget,
)

# Internal Imports
from init import DESIGN_DIR, Propeller
from gui.design_tab import DesignTab
from gui.optimise_tab import OptimiseTab
from gui.setup_tab import SetupTab
from gui.workers import LogStream

APC_DIR = Path(__file__).resolve().parent.parent / "apc_prop_data" / "propgeom_data"


class MainWindow(QMainWindow):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        ## Tabs
        self.setup_tab = SetupTab(session)
        self.design_tab = DesignTab(session)
        self.optimise_tab = OptimiseTab(session, self.setup_tab)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.design_tab, "Design")
        self.tabs.addTab(self.optimise_tab, "Optimise")
        self.setCentralWidget(self.tabs)
        self.optimise_tab.result_adopted.connect(self._show_design)

        ## Log: everything printed (solver tables, BEMT output, warnings) also appears here
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [150], Qt.Vertical)

        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = LogStream(self._streams[0], self)
        sys.stderr = LogStream(self._streams[1], self)
        for stream in (sys.stdout, sys.stderr):
            stream.text.connect(self._append_log)

        self._build_menus()

        ## Status bar
        self.busy_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.busy_label)
        session.busy_changed.connect(self._busy_changed)
        session.dirty_changed.connect(self._update_title)
        session.propeller_changed.connect(self._update_title)
        self._update_title()

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("&New default propeller", QKeySequence.New, self.new_default),
            ("&Open...", QKeySequence.Open, self.open),
            ("&Import APC geometry...", None, self.import_apc),
            (None, None, None),
            ("&Save", QKeySequence.Save, self.save),
            ("Save &As...", QKeySequence.SaveAs, self.save_as),
            (None, None, None),
            ("&Quit", QKeySequence.Quit, self.close),
        ):
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.log_dock.toggleViewAction())
        clear = QAction("Clear log", self)
        clear.triggered.connect(self.log.clear)
        view_menu.addAction(clear)

    ## Window state
    def _update_title(self, *_):
        self.setWindowTitle(f"Proptimiser - {self.session.name}{' *' if self.session.dirty else ''}")

    def _busy_changed(self, busy, task_name):
        self.busy_label.setText(f"{task_name}..." if busy else "Ready")

    def _append_log(self, text):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _show_design(self):
        self.tabs.setCurrentWidget(self.design_tab)

    ## File actions
    def _discard_ok(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", f"Save changes to {self.session.name} first?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Save:
            return self.save()
        return answer == QMessageBox.Discard

    def _replace(self, prop, path, dirty):
        self.session.path = path
        self.session.set_propeller(prop, dirty=dirty)

    def new_default(self):
        if self._discard_ok():
            self._replace(Propeller.default(), None, dirty=False)

    def open(self):
        if not self._discard_ok():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open design", str(DESIGN_DIR), "Design (*.yaml)")
        if not path:
            return
        try:
            prop = Propeller.load(path)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
            QMessageBox.critical(self, "Open failed", f"{Path(path).name}: {e}")
            return
        self._replace(prop, Path(path), dirty=False)

    def import_apc(self):
        if not self._discard_ok():
            return
        start = APC_DIR if APC_DIR.exists() else Path.cwd()
        path, _ = QFileDialog.getOpenFileName(self, "Import APC geometry", str(start), "APC geometry (*.PE0 *.pe0)")
        if not path:
            return
        current = self.session.prop
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                prop = Propeller.from_apc(path, inputs=current.inputs, blade=current.blade)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Import failed", f"{Path(path).name}: {e}")
            return
        prop.name = Path(path).stem
        self._replace(prop, None, dirty=True)

    def save(self) -> bool:
        if self.session.path is None:
            return self.save_as()
        return self._write(self.session.path)

    def save_as(self) -> bool:
        default = DESIGN_DIR / f"{self.session.name}.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save design", str(default), "Design (*.yaml)")
        return bool(path) and self._write(Path(path))

    def _write(self, path) -> bool:
        try:
            path = self.session.prop.save(path, overwrite=True)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        self.session.path = path
        self.session.set_dirty(False)
        self._update_title()
        return True

    ## Closing
    def closeEvent(self, event):
        if self.optimise_tab.is_running():
            if QMessageBox.question(self, "Optimisation running", "Stop the optimisation and quit?") != QMessageBox.Yes:
                event.ignore()
                return
            self.optimise_tab.stop()
        if not self._discard_ok():
            event.ignore()
            return

        self.design_tab.close_views()
        sys.stdout, sys.stderr = self._streams
        event.accept()
