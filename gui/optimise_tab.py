## Optimise tab: run the optimiser with live plots, per-condition progress and Start/Stop

# External Imports
import threading
import time
import traceback
import warnings
from datetime import datetime

import jax
import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# Internal Imports
import proptimiser
import results
from init import DESIGN_DIR
from gui.widgets import LivePlotCanvas, make_table, fill_table, fit_table_height
from gui.workers import Task

PROGRESS_COLUMNS = (
    ("Cond", "{}"), ("Omega [rad/s]", "{:.1f}"), ("Thrust [N]", "{:.3f}"), ("Target [N]", "{:.3f}"),
    ("T err [%]", "{:+.2f}"), ("Motor power [W]", "{:.1f}"), ("P / P ideal", "{:.3f}"), ("Confidence", "{:.3f}"),
)

FINAL_STATUS = {
    "converged": "Converged",
    "stopped": "Stopped by user",
    "max_iter": "Reached the iteration limit",
    "failed": "Solver stopped early",
}

JAX_TRACE_DIR = "/tmp/jax_trace"


class OptimiseTab(QWidget):
    '''
    Optimises the session's propeller on a background thread. Progress reports (proptimiser.Progress plus a
    FlowSnapshot per condition, computed off the Qt thread) drive the live plot and the progress table.
    '''
    result_adopted = Signal()       # the optimised design became the session propeller

    def __init__(self, session, setup_tab, parent=None):
        super().__init__(parent)
        self.session = session
        self.setup_tab = setup_tab
        self.stop_event = None
        self.running = False
        self.result = None
        self.latest = None          # (Progress, [FlowSnapshot per condition])
        self.started = 0.0

        ## Controls
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        self.status = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m iterations")
        self.progress_bar.setMaximumWidth(260)
        self.elapsed = QLabel()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick)

        top = QHBoxLayout()
        top.addWidget(self.start_button)
        top.addWidget(self.stop_button)
        top.addWidget(self.status, 1)
        top.addWidget(self.progress_bar)
        top.addWidget(self.elapsed)

        self.condition = QComboBox()
        self.condition.currentIndexChanged.connect(self._redraw)
        self.design_space = QLabel()
        self.design_space.setWordWrap(True)
        self.use_button = QPushButton("Use as current design")
        self.use_button.clicked.connect(self.use_result)
        self.save_button = QPushButton("Save result...")
        self.save_button.clicked.connect(self.save_result)
        for button in (self.use_button, self.save_button):
            button.setEnabled(False)

        second = QHBoxLayout()
        second.addWidget(QLabel("Plot condition"))
        second.addWidget(self.condition)
        second.addWidget(self.design_space, 1)
        second.addWidget(self.use_button)
        second.addWidget(self.save_button)

        ## Live plot over the per-condition progress table
        self.canvas = LivePlotCanvas()
        self.canvas.message("Press Start to optimise the current design")
        self.table = make_table([h for h, _ in PROGRESS_COLUMNS])
        fit_table_height(self.table)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(second)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table)

        session.propeller_changed.connect(self._propeller_changed)
        session.busy_changed.connect(self._busy_changed)
        self._propeller_changed(session.prop)

    ## Session updates
    def _propeller_changed(self, prop):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                space = proptimiser.DesignSpace.build(prop)
                text = "Design variables: " + ", ".join(f"{name} {len(idx)}" for name, idx in space.free)
            except ValueError as e:
                text = str(e)
        notes = "; ".join(str(w.message) for w in caught)
        self.design_space.setText(text + (f"  ({notes})" if notes else ""))

        if not self.running:
            self._fill_conditions(prop.inputs.n_cond)

    def _busy_changed(self, busy, _):
        self.start_button.setEnabled(not busy)

    def _fill_conditions(self, n_cond):
        selected = max(self.condition.currentIndex(), 0)
        self.condition.blockSignals(True)
        self.condition.clear()
        self.condition.addItems([f"C{i}" for i in range(n_cond)])
        self.condition.setCurrentIndex(min(selected, n_cond - 1))
        self.condition.blockSignals(False)

    ## Running
    def is_running(self) -> bool:
        return self.running

    def start(self):
        if self.running:
            return
        if self.setup_tab.has_pending_changes():
            answer = QMessageBox.question(
                self, "Unapplied setup changes", "Apply the changes in the Setup tab before optimising?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                self.setup_tab.apply()
                if self.setup_tab.has_pending_changes():
                    QMessageBox.warning(self, "Setup", "The Setup changes could not be applied; see the Setup tab.")
                    return
        if not self.session.try_begin("Optimising"):
            QMessageBox.information(self, "Busy", f"Wait for '{self.session.task_name}' to finish first.")
            return

        prop = self.session.prop
        n_cond = prop.inputs.n_cond
        stop_event = self.stop_event = threading.Event()

        self.running = True
        self.result = None
        self.latest = None
        self.started = time.time()
        self.elapsed_timer.start(1000)
        self._tick()
        self._fill_conditions(n_cond)
        self.progress_bar.setRange(0, prop.inputs.opt_max_iter)
        self.progress_bar.setValue(0)
        self.status.setText("Compiling the solver (the first iteration can take a while)...")
        self.table.setRowCount(0)
        fit_table_height(self.table)
        self.canvas.message("Waiting for the first iteration...")
        self.stop_button.setEnabled(True)
        for button in (self.use_button, self.save_button):
            button.setEnabled(False)

        def work(report):
            def callback(progress):
                # Plot data is computed here, on the optimiser thread, so the Qt thread only draws
                try:
                    snaps = [results.plot_snapshot(progress.condition(i)) for i in range(n_cond)]
                except Exception:
                    traceback.print_exc()
                    snaps = None
                report((progress, snaps))

            if prop.inputs.jax_tracing:
                jax.profiler.start_trace(JAX_TRACE_DIR)
            try:
                return proptimiser.run_optimization(prop, callback=callback, stop_event=stop_event)
            finally:
                if prop.inputs.jax_tracing:
                    jax.profiler.stop_trace()
                    print(f"JAX profiler trace saved to {JAX_TRACE_DIR}")

        task = Task(work, self)
        task.progress.connect(self._on_progress)
        task.done.connect(self._on_done)
        task.failed.connect(self._on_failed)
        task.finished.connect(self._on_finished)
        task.start()

    def stop(self):
        if self.running and self.stop_event is not None:
            self.stop_event.set()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the current iteration...")

    def _tick(self):
        self.elapsed.setText(f"{time.time() - self.started:.0f} s")

    def _on_progress(self, update):
        progress, snaps = update
        self.latest = update
        self.progress_bar.setValue(progress.iteration)

        summary = (f"objective {progress.objective:.5f} x ideal power, "
                   + ("feasible" if progress.violation <= 0 else f"constraint violation {100 * progress.violation:.2f}%"))
        if progress.status != "running":
            detail = f": {progress.message}" if progress.message else ""
            self.status.setText(f"{FINAL_STATUS[progress.status]}{detail} after {progress.iteration} iterations "
                                f"({progress.elapsed:.1f} s), {summary}")
        elif not self.stop_event.is_set():
            self.status.setText(f"Running: iteration {progress.iteration}, {summary}, change {progress.error:.1e}")

        fill_table(self.table, [
            (f"C{i}", progress.omega[i], progress.thrust[i], progress.thrust_target[i], progress.thrust_err_pct[i],
             progress.power[i], progress.power_ratio[i],
             progress.confidence[i] if np.isfinite(progress.confidence[i]) else "-")
            for i in range(len(progress.omega))
        ], formats=[fmt for _, fmt in PROGRESS_COLUMNS])
        fit_table_height(self.table)
        self._redraw()

    def _redraw(self, *_):
        if self.latest is None or self.latest[1] is None:
            return
        idx = max(self.condition.currentIndex(), 0)
        snaps = self.latest[1]
        if idx < len(snaps):
            self.canvas.show_snapshot(snaps[idx])

    def _on_done(self, prop_opt):
        self.result = prop_opt
        for button in (self.use_button, self.save_button):
            button.setEnabled(True)

    def _on_failed(self, text):
        self.status.setText("Failed (see the log)")
        QMessageBox.critical(self, "Optimisation failed", text.strip().splitlines()[-1])

    def _on_finished(self):
        self.running = False
        self.elapsed_timer.stop()
        self.stop_button.setEnabled(False)
        self.session.end()

    ## Results
    def use_result(self):
        if self.result is None:
            return
        self.session.set_propeller(self.result)
        self.result_adopted.emit()

    def save_result(self):
        if self.result is None:
            return
        default = DESIGN_DIR / f"{self.result.name or 'prop'}_{datetime.now().strftime('%Y%m%d-%H%M')}.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save optimised design", str(default), "Design (*.yaml)")
        if not path:
            return
        try:
            self.result.save(path, overwrite=True)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
