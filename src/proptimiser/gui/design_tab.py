## Design tab: evaluate a blade, inspect its sectional data, edit the planform, view it in 3D and export to Creo

# External Imports
import csv
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

# Internal Imports
from .. import geometry
from .. import results
from .. import structure
from ..init import DESIGN_DIR
from .setup_tab import MODEL_CHOICES
from .widgets import FloatEdit, LivePlotCanvas, MplCanvas, make_table, fill_table, fit_table_height
from .workers import Task

PERFORMANCE_COLUMNS = (
    ("Cond", "{}"), ("Omega [rad/s]", "{:.1f}"), ("Thrust [N]", "{:.3f}"), ("Target [N]", "{:.3f}"),
    ("T err [%]", "{:+.2f}"), ("Motor power [W]", "{:.1f}"), ("Shaft power [W]", "{:.1f}"),
    ("Ideal / shaft power", "{:.3f}"),
)

# (header, diagnostics_data key, scale, format)
SECTION_COLUMNS = (
    ("r [mm]", "r", 1000.0, "{:.1f}"), ("c [mm]", "c", 1000.0, "{:.2f}"), ("beta [deg]", "beta_deg", 1.0, "{:.2f}"),
    ("phi [deg]", "phi_deg", 1.0, "{:.2f}"), ("alpha [deg]", "alpha_deg", 1.0, "{:.2f}"),
    ("V_R [m/s]", "V_R", 1.0, "{:.2f}"), ("Mach", "Mach", 1.0, "{:.3f}"), ("Re", "Re", 1.0, "{:.3g}"),
    ("CL", "CL", 1.0, "{:.4f}"), ("CD", "CD", 1.0, "{:.5f}"), ("dT [N/m]", "dT", 1.0, "{:.3f}"),
    ("dQ [Nm/m]", "dQ", 1.0, "{:.4f}"),
)

# (header, structural_data key, scale, format): the largest value over the azimuth, or that of the chosen azimuth
STRUCTURE_COLUMNS = (
    ("r [mm]", "r", 1000.0, "{:.1f}"), ("N [N]", "N", 1.0, "{:.1f}"), ("Shear x [N]", "V_x", 1.0, "{:.2f}"),
    ("Shear z [N]", "V_z", 1.0, "{:.2f}"), ("Flap M [N m]", "M_x", 1.0, "{:.4f}"), ("Edge M [N m]", "M_z", 1.0, "{:.4f}"),
    ("Torque [N m]", "T", 1.0, "{:.4f}"), ("Von Mises [MPa]", "sigma_vm", 1e-6, "{:.2f}"),
    ("Shear stress [MPa]", "tau", 1e-6, "{:.2f}"), ("Safety factor", "safety_factor", 1.0, "{:.2f}"),
)

STALE_STYLE = "color: #d35400;"


def blade_summary(prop) -> str:
    b, i = prop.blade, prop.inputs
    stations = ", ".join(f"{r * 1000:.0f}" for r in b.r_foils[2:])
    return (
        f"{b.N} blades, tip radius {b.R_tip * 1000:.1f} mm\n"
        f"Hub to {b.R_hub * 1000:.1f} mm, blend to {b.R_aero * 1000:.1f} mm\n"
        f"Sections: {', '.join(b.airfoils)}\n  at r = {stations} mm\n"
        f"{i.n_r} radial x {i.n_psi} azimuthal stations\n"
        f"{i.n_cond} flight condition{'s' if i.n_cond != 1 else ''}, "
        f"{MODEL_CHOICES['coeff_method'][i.coeff_method]} coefficients"
    )


class BlendDialog(QDialog):
    '''
    Picks saved propellers and weights for results.propgeomblender.
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blend saved propellers")
        self.names = sorted(p.stem for p in DESIGN_DIR.glob("*.yaml"))

        self.table = make_table(["Propeller", "Weight"])
        self.table.setRowCount(len(self.names))
        self.weights = []
        for i, name in enumerate(self.names):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            self.table.setItem(i, 0, item)
            weight = FloatEdit(1.0)
            self.table.setCellWidget(i, 1, weight)
            self.weights.append(weight)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Chord, twist and CST sections are averaged with the normalised weights.\n"
                                "All propellers must share the radial grid and number of sections;\n"
                                "the result uses the last propeller's settings."))
        layout.addWidget(self.table)
        layout.addWidget(buttons)
        self.resize(420, 360)

    def selection(self):
        chosen = [(name, w.value()) for i, (name, w) in enumerate(zip(self.names, self.weights))
                  if self.table.item(i, 0).checkState() == Qt.Checked]
        if not chosen:
            raise ValueError("Select at least one propeller")
        names, weights = zip(*chosen)
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            raise ValueError("Weights must sum to a positive value")
        return list(names), weights / weights.sum()


class DesignTab(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.evaluation = None          # dict(rows, snaps, diags) of the last evaluation
        self.editors = []
        self.interactor = None
        self.mesh_stale = True
        self.export_geom = None         # (prop, blade_export_geometry output) of the last 3D build

        ## Left panel: propeller and evaluation
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        blend_button = QPushButton("Blend saved propellers...")
        blend_button.clicked.connect(self.blend)
        prop_box = QGroupBox("Propeller")
        prop_layout = QVBoxLayout(prop_box)
        prop_layout.addWidget(self.summary)
        prop_layout.addWidget(blend_button)

        self.eval_button = QPushButton("Evaluate all conditions")
        self.eval_button.clicked.connect(self.evaluate)
        self.eval_status = QLabel("Not evaluated")
        self.eval_status.setWordWrap(True)
        self.condition = QComboBox()
        self.condition.currentIndexChanged.connect(self._draw_condition)
        eval_box = QGroupBox("Performance")
        eval_layout = QFormLayout(eval_box)
        eval_layout.addRow(self.eval_button)
        eval_layout.addRow(self.eval_status)
        eval_layout.addRow("Plot condition", self.condition)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(prop_box)
        left_layout.addWidget(eval_box)
        left_layout.addStretch()
        left.setMaximumWidth(340)

        ## Right: performance table over the sub-tabs
        self.perf_table = make_table([h for h, _ in PERFORMANCE_COLUMNS])
        fit_table_height(self.perf_table)

        self.views = QTabWidget()
        self.perf_canvas = LivePlotCanvas()
        self.diag_canvas = MplCanvas(figsize=(14, 10))
        self.views.addTab(self.perf_canvas, "Performance")
        self.views.addTab(self.diag_canvas, "Diagnostics")
        self.views.addTab(self._build_sections_view(), "Sectional data")
        self.views.addTab(self._build_structural_view(), "Structural")
        self.views.addTab(self._build_planform_view(), "Planform")
        self.export_view = self._build_export_view()
        self.views.addTab(self.export_view, "3D && Export")
        self.views.currentChanged.connect(self._view_changed)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.perf_table)
        right_layout.addWidget(self.views, 1)

        layout = QHBoxLayout(self)
        layout.addWidget(left)
        layout.addWidget(right, 1)

        for canvas in (self.perf_canvas, self.diag_canvas, self.struct_canvas):
            canvas.message("Evaluate the design to see results")

        session.propeller_changed.connect(self._propeller_changed)
        session.busy_changed.connect(self._busy_changed)
        self._propeller_changed(session.prop)

    ## Sub-tab construction
    def _build_sections_view(self):
        self.section_table = make_table([h for h, *_ in SECTION_COLUMNS])
        copy_button = QPushButton("Copy table")
        copy_button.clicked.connect(self.copy_sections)
        csv_button = QPushButton("Export CSV...")
        csv_button.clicked.connect(self.export_sections)
        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Azimuth-averaged values at each radial station of the plotted condition"))
        buttons.addStretch()
        buttons.addWidget(copy_button)
        buttons.addWidget(csv_button)

        view = QWidget()
        layout = QVBoxLayout(view)
        layout.addWidget(self.section_table)
        layout.addLayout(buttons)
        return view

    def _build_structural_view(self):
        self.struct_canvas = MplCanvas(figsize=(14, 8))
        self.struct_summary = QLabel("Evaluate the design to see the structural loads")
        self.struct_summary.setWordWrap(True)
        self.struct_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Motion beyond steady rotation (structure.Kinematics): all zero for a steady operating point
        self.omega_dot = FloatEdit(0.0)
        self.pitch_rate = FloatEdit(0.0)
        self.pitch_accel = FloatEdit(0.0)
        self.body_rates = [FloatEdit(0.0) for _ in range(3)]
        self.omega_dot.setToolTip("Shaft spin acceleration")
        self.pitch_rate.setToolTip("Rate of change of the blade pitch about its axis (a constant-speed unit at work)")
        self.pitch_accel.setToolTip("Acceleration of the pitch change")
        for name, edit in zip("pqr", self.body_rates):
            edit.setToolTip(f"Airframe angular rate {name} about the hub axes (x at azimuth 0, y at 90 deg, z along the thrust)")
        self.azimuth_choice = QComboBox()
        self.azimuth_choice.addItem("Envelope")
        self.azimuth_choice.currentIndexChanged.connect(self._draw_structural_current)
        self.recalc_button = QPushButton("Recalculate")
        self.recalc_button.setToolTip("Recalculates the structural loads of the evaluated design with the motion set here. "
                                      "Change the material in Setup, then evaluate again.")
        self.recalc_button.clicked.connect(self.recalculate_structure)

        controls = QFormLayout()
        controls.addRow("Spin acceleration [rad/s^2]", self.omega_dot)
        controls.addRow("Pitch rate [deg/s]", self.pitch_rate)
        controls.addRow("Pitch acceleration [deg/s^2]", self.pitch_accel)
        body = QHBoxLayout()
        for name, edit in zip("pqr", self.body_rates):
            edit.setMaximumWidth(60)
            body.addWidget(QLabel(name))
            body.addWidget(edit)
        controls.addRow("Airframe rates [deg/s]", body)
        controls.addRow("Blade azimuth", self.azimuth_choice)
        controls.addRow(self.recalc_button)
        controls_box = QGroupBox("Motion")
        controls_box.setLayout(controls)
        controls_box.setMaximumWidth(330)

        self.struct_table = make_table([h for h, *_ in STRUCTURE_COLUMNS])
        copy_button = QPushButton("Copy table")
        copy_button.clicked.connect(self.copy_structure)
        csv_button = QPushButton("Export CSV...")
        csv_button.clicked.connect(self.export_structure)
        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Blade loads: largest over the azimuth (or the chosen azimuth), per radial station"))
        buttons.addStretch()
        buttons.addWidget(copy_button)
        buttons.addWidget(csv_button)

        top = QHBoxLayout()
        top.addWidget(controls_box)
        top.addWidget(self.struct_summary, 1)
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.addLayout(top)
        layout.addWidget(self.struct_canvas, 3)
        layout.addWidget(self.struct_table, 1)
        layout.addLayout(buttons)
        return view

    def _build_planform_view(self):
        self.planform_canvas = MplCanvas(figsize=(14, 6))
        self.cp_spin = QSpinBox()
        self.cp_spin.setRange(4, 20)
        self.cp_spin.setValue(8)
        self.cp_spin.valueChanged.connect(self._build_editors)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._build_editors)
        apply_button = QPushButton("Apply to design")
        apply_button.clicked.connect(self.apply_planform)
        self.planform_status = QLabel("Drag the red control points; the blend zone is derived from the hub values")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Control points"))
        controls.addWidget(self.cp_spin)
        controls.addWidget(reset_button)
        controls.addWidget(apply_button)
        controls.addWidget(self.planform_status, 1)

        view = QWidget()
        layout = QVBoxLayout(view)
        layout.addWidget(self.planform_canvas)
        layout.addLayout(controls)
        return view

    def _build_export_view(self):
        self.view_3d = QWidget()
        self.view_3d_layout = QVBoxLayout(self.view_3d)
        self.view_3d_layout.setContentsMargins(0, 0, 0, 0)
        self.view_3d_placeholder = QLabel("The 3D view loads when this tab is opened")
        self.view_3d_placeholder.setAlignment(Qt.AlignCenter)
        self.view_3d_layout.addWidget(self.view_3d_placeholder)

        self.refresh_3d_button = QPushButton("Update 3D view")
        self.refresh_3d_button.clicked.connect(self.update_3d)

        self.export_name = QLineEdit()
        self.export_dir = QLineEdit(str(geometry.EXPORT_DIR))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_export_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.export_dir)
        dir_row.addWidget(browse)

        self.export_mold = QCheckBox("Aerodynamic mold (outer skin)")
        self.export_insert = QCheckBox("Core insert (skin offset inward)")
        self.export_parting = QCheckBox("Parting surface")
        for box in (self.export_mold, self.export_insert, self.export_parting):
            box.setChecked(True)
        self.laminate = FloatEdit(1.5)
        self.n_camber = QSpinBox()
        self.n_camber.setRange(10, 1000)
        self.n_camber.setValue(100)
        self.x_margin = FloatEdit(5.0)
        self.y_margin = FloatEdit(5.0)
        self.export_button = QPushButton("Export to Creo (.ibl)")
        self.export_button.clicked.connect(self.export_creo)

        form = QFormLayout()
        form.addRow(self.refresh_3d_button)
        form.addRow("File name prefix", self.export_name)
        form.addRow("Folder", dir_row)
        form.addRow(self.export_mold)
        form.addRow(self.export_insert)
        form.addRow("Laminate thickness [mm]", self.laminate)
        form.addRow(self.export_parting)
        form.addRow("Camber points", self.n_camber)
        form.addRow("Chordwise margin [mm]", self.x_margin)
        form.addRow("Spanwise margin [mm]", self.y_margin)
        form.addRow(self.export_button)
        panel = QGroupBox("Creo export")
        panel.setLayout(form)
        panel.setMaximumWidth(380)

        view = QWidget()
        layout = QHBoxLayout(view)
        layout.addWidget(self.view_3d, 1)
        layout.addWidget(panel)
        return view

    ## Propeller changes
    def _propeller_changed(self, prop):
        self.summary.setText(f"<b>{prop.name or 'untitled'}</b><br>" + blade_summary(prop).replace("\n", "<br>"))
        self.export_name.setText(prop.name or "propeller")
        if self.evaluation is not None:
            self._set_stale("Design changed: re-evaluate to update the results")
        self.mesh_stale = True
        self._build_editors()
        if self.views.currentWidget() is self.export_view:
            self.update_3d()

    def _set_stale(self, message):
        self.eval_status.setStyleSheet(STALE_STYLE)
        self.eval_status.setText(message)

    def _busy_changed(self, busy, _):
        for button in (self.eval_button, self.refresh_3d_button, self.export_button, self.recalc_button):
            button.setEnabled(not busy)

    def _view_changed(self, index):
        if self.views.widget(index) is self.export_view:
            self._ensure_interactor()
            if self.mesh_stale:
                self.update_3d()

    ## Evaluation
    def _kinematics(self):
        '''
        structure.Kinematics from the Motion fields (deg and rad/s in the GUI, SI inside); ValueError on bad input.
        '''
        degrees = np.pi / 180.0
        return structure.Kinematics(
            omega_dot=np.asarray(self.omega_dot.value()), pitch_rate=np.asarray(self.pitch_rate.value() * degrees),
            pitch_accel=np.asarray(self.pitch_accel.value() * degrees),
            body_rates=np.array([edit.value() for edit in self.body_rates]) * degrees)

    def evaluate(self):
        try:
            kin = self._kinematics()
        except ValueError as e:
            QMessageBox.warning(self, "Motion", str(e))
            return
        if not self.session.try_begin("Evaluating flight conditions"):
            return
        prop = self.session.prop
        material = prop.material()
        started = time.time()
        self.eval_status.setStyleSheet("")
        self.eval_status.setText("Evaluating... (the first run compiles the model, which can take a minute)")

        def work(report):
            prop_warm, params_list, rows = results.evaluate_conditions(prop)
            snaps = [results.plot_snapshot(p) for p in params_list]
            diags = [results.diagnostics_data(p) for p in params_list]
            structural = [results.structural_data(p, material, kin) for p in params_list]
            return prop, prop_warm, rows, snaps, diags, (params_list, material, structural), time.time() - started

        task = Task(work, self)
        task.done.connect(self._evaluated)
        task.failed.connect(self._evaluate_failed)
        task.finished.connect(self.session.end)
        task.start()

    def _evaluated(self, result):
        source, prop_warm, rows, snaps, diags, (params_list, material, structural), seconds = result
        self.evaluation = dict(rows=rows, snaps=snaps, diags=diags, params=params_list, material=material,
                               structural=structural)

        if self.session.prop is source:
            # Keep the solved induced velocity as the warm start; the design itself is unchanged
            self.session.set_propeller(prop_warm, dirty=self.session.dirty, notify=False)
            self.eval_status.setStyleSheet("")
            self.eval_status.setText(f"Evaluated in {seconds:.1f} s")
        else:
            self._set_stale("These results are for an earlier design: re-evaluate")

        fill_table(self.perf_table, [
            (f"C{i}", r["omega"], r["thrust"], r["thrust_target"], r["thrust_err_pct"],
             r["motor_power"], r["shaft_power"], r["ideal_ratio"])
            for i, r in enumerate(rows)
        ], formats=[fmt for _, fmt in PERFORMANCE_COLUMNS])
        fit_table_height(self.perf_table)

        selected = max(self.condition.currentIndex(), 0)
        self.condition.blockSignals(True)
        self.condition.clear()
        self.condition.addItems([f"C{i}: {r['thrust_target']:.1f} N at {v:.1f} m/s"
                                 for i, (r, v) in enumerate(zip(rows, np.asarray(source.inputs.V_x)))])
        self.condition.setCurrentIndex(min(selected, len(rows) - 1))
        self.condition.blockSignals(False)
        self._draw_condition(self.condition.currentIndex())

    def _evaluate_failed(self, text):
        self._set_stale("Evaluation failed (see the log)")
        QMessageBox.critical(self, "Evaluation failed", text.strip().splitlines()[-1])

    def _draw_condition(self, idx):
        if self.evaluation is None or idx < 0:
            return
        snap, diag = self.evaluation["snaps"][idx], self.evaluation["diags"][idx]

        self.perf_canvas.show_snapshot(snap)

        results.debug_diagnostics_plot(None, idx, show=False, fig=self.diag_canvas.figure, usetex=False, data=diag)
        self.diag_canvas.draw()

        fill_table(self.section_table, [
            [diag[key][i] * scale for _, key, scale, _ in SECTION_COLUMNS] for i in range(len(diag["r"]))
        ], formats=[fmt for *_, fmt in SECTION_COLUMNS])

        n_psi = len(self.evaluation["structural"][idx]["psi"])
        if self.azimuth_choice.count() != n_psi + 1:
            self.azimuth_choice.blockSignals(True)
            self.azimuth_choice.clear()
            self.azimuth_choice.addItem("Envelope")
            self.azimuth_choice.addItems([f"{np.degrees(a):.0f} deg" for a in self.evaluation["structural"][idx]["psi"]])
            self.azimuth_choice.blockSignals(False)
        self._draw_structural(idx)

    ## Structural loads
    def _draw_structural_current(self, *_):
        self._draw_structural(self.condition.currentIndex())

    def _draw_structural(self, idx):
        if self.evaluation is None or idx < 0:
            return
        data = self.evaluation["structural"][idx]
        psi_index = None if self.azimuth_choice.currentIndex() <= 0 else self.azimuth_choice.currentIndex() - 1
        results.structural_plot(data, idx, show=False, fig=self.struct_canvas.figure, psi_index=psi_index)
        self.struct_canvas.draw()
        self.struct_summary.setText(results.structural_summary(data, idx))

        def column(key):
            values = np.asarray(data[key])
            if key == "r":
                return values
            if key == "safety_factor":
                safety = values.min(axis=0) if psi_index is None else values[psi_index]
                return np.where(safety > 1e3, np.nan, safety)           # the unloaded tip
            if key in ("sigma_vm", "tau"):
                return values.max(axis=0) if psi_index is None else values[psi_index]
            return results.structural_peak(values, psi_index)

        columns = [column(key) * scale for _, key, scale, _ in STRUCTURE_COLUMNS]
        fill_table(self.struct_table, [list(row) for row in zip(*columns)],
                   formats=[fmt for *_, fmt in STRUCTURE_COLUMNS])

    def recalculate_structure(self):
        if self.evaluation is None:
            QMessageBox.information(self, "Structural loads", "Evaluate the design first")
            return
        try:
            kin = self._kinematics()
        except ValueError as e:
            QMessageBox.warning(self, "Motion", str(e))
            return
        if not self.session.try_begin("Recalculating the structural loads"):
            return
        params_list, material = self.evaluation["params"], self.evaluation["material"]

        def work(report):
            return [results.structural_data(p, material, kin) for p in params_list]

        task = Task(work, self)
        task.done.connect(self._structure_recalculated)
        task.failed.connect(self._task_failed)
        task.finished.connect(self.session.end)
        task.start()

    def _structure_recalculated(self, structural):
        if self.evaluation is None:
            return
        self.evaluation["structural"] = structural
        self._draw_structural(self.condition.currentIndex())

    def _structure_rows(self):
        headers = [h for h, *_ in STRUCTURE_COLUMNS]
        rows = [[self.struct_table.item(i, j).text() for j in range(len(headers))]
                for i in range(self.struct_table.rowCount())]
        return headers, rows

    def copy_structure(self):
        headers, rows = self._structure_rows()
        QApplication.clipboard().setText("\n".join("\t".join(row) for row in [headers] + rows))

    def export_structure(self):
        if self.evaluation is None:
            return
        default = geometry.EXPORT_DIR / f"{self.session.name}_C{self.condition.currentIndex()}_structure.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export structural loads", str(default), "CSV (*.csv)")
        if not path:
            return
        headers, rows = self._structure_rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([headers] + rows)

    ## Sectional data
    def _section_rows(self):
        headers = [h for h, *_ in SECTION_COLUMNS]
        rows = [[self.section_table.item(i, j).text() for j in range(len(headers))]
                for i in range(self.section_table.rowCount())]
        return headers, rows

    def copy_sections(self):
        headers, rows = self._section_rows()
        QApplication.clipboard().setText("\n".join("\t".join(row) for row in [headers] + rows))

    def export_sections(self):
        if self.evaluation is None:
            return
        default = geometry.EXPORT_DIR / f"{self.session.name}_C{self.condition.currentIndex()}_sections.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export sectional data", str(default), "CSV (*.csv)")
        if not path:
            return
        headers, rows = self._section_rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([headers] + rows)

    ## Planform editing
    def _build_editors(self):
        for editor in self.editors:
            editor.disconnect()
        fig = self.planform_canvas.figure
        fig.clear()
        ax_c, ax_b = fig.subplots(1, 2)

        prop = self.session.prop
        blade = prop.blade
        r_mm = np.asarray(prop.r) * 1000
        n = self.cp_spin.value()
        self.editors = [
            results.DraggableSplineEditor(r_mm, np.asarray(prop.c) * 1000, num_cps=n, name="Chord (mm)", ax=ax_c),
            results.DraggableSplineEditor(r_mm, np.degrees(np.asarray(prop.beta)), num_cps=n, name="Twist (deg)", ax=ax_b),
        ]
        for ax, editor in zip((ax_c, ax_b), self.editors):
            ax.axvspan(blade.R_hub * 1000, blade.R_aero * 1000, color="0.88", zorder=0, label="Blend zone (from hub)")
            ax.set_xlabel("Radius (mm)")
            # Headroom above the data so points can be dragged upwards
            lo, hi = min(0.0, float(editor.y_dense.min())), max(float(editor.y_dense.max()), 1e-6)
            span = hi - lo
            ax.set_ylim(lo - (0.1 * span if lo < 0 else 0.0), hi + 0.5 * span)
            ax.legend(loc="best", fontsize=8)
        self.planform_canvas.draw()

    def apply_planform(self):
        prop = self.session.prop
        blade = prop.blade
        c = self.editors[0].get_final_data() / 1000.0
        beta = np.radians(self.editors[1].get_final_data())

        # Same blend zone as the optimiser and the CAD export, then the design bounds
        c, beta = geometry.blend_planform(np.asarray(prop.r), c, beta, blade.R_hub, blade.R_aero,
                                          blade.hub_chord, np.radians(blade.hub_twist_deg))
        c, beta = np.asarray(c), np.asarray(beta)
        c_clipped = np.clip(c, *blade.c_bounds)
        beta_clipped = np.clip(beta, *np.radians(blade.beta_bounds_deg))
        clipped = not (np.allclose(c, c_clipped) and np.allclose(beta, beta_clipped))

        self.session.set_propeller(prop.with_state(c=c_clipped, beta=beta_clipped))
        self.planform_status.setText("Applied" + (" (clipped to the chord/twist bounds in Setup)" if clipped else ""))

    ## 3D view and export
    def _ensure_interactor(self):
        if self.interactor is not None or self.view_3d_placeholder is None:
            return
        try:
            from pyvistaqt import QtInteractor
            self.interactor = QtInteractor(self.view_3d)
        except Exception as e:      # missing package or no OpenGL context
            self.view_3d_placeholder.setText(f"3D view unavailable: {e}")
            self.view_3d_placeholder = None
            return
        self.interactor.set_background("#1e1e2f")
        self.view_3d_layout.removeWidget(self.view_3d_placeholder)
        self.view_3d_placeholder.deleteLater()
        self.view_3d_placeholder = None
        self.view_3d_layout.addWidget(self.interactor.interactor)

    def update_3d(self):
        if not self.session.try_begin("Building the 3D blade"):
            return
        prop = self.session.prop

        def work(report):
            geom = geometry.blade_export_geometry(prop)
            r, c, beta, pitch_axis, coords, offsets = geom
            mesh = geometry.build_propeller_mesh(r, c, beta, coords, pitch_axis=pitch_axis, num_blades=prop.blade.N,
                                                 offsets=offsets)
            return prop, geom, mesh

        task = Task(work, self)
        task.done.connect(self._show_mesh)
        task.failed.connect(self._task_failed)
        task.finished.connect(self.session.end)
        task.start()

    def _show_mesh(self, result):
        prop, geom, mesh = result
        self.export_geom = (prop, geom)
        self.mesh_stale = self.session.prop is not prop
        if self.interactor is not None:
            self.interactor.clear()
            self.interactor.add_mesh(mesh, color="silver", smooth_shading=True)
            self.interactor.add_axes()
            self.interactor.reset_camera()

    def _browse_export_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Export folder", self.export_dir.text())
        if folder:
            self.export_dir.setText(folder)

    def export_creo(self):
        prefix = self.export_name.text().strip() or self.session.name
        out_dir = Path(self.export_dir.text().strip() or geometry.EXPORT_DIR)
        targets = [(kind, f"{prefix}_{suffix}") for kind, suffix, box in (
            ("mold", "mold", self.export_mold), ("insert", "insert", self.export_insert),
            ("parting", "parting_surf", self.export_parting)) if box.isChecked()]
        if not targets:
            return
        try:
            laminate = self.laminate.value() / 1000.0
            x_margin, y_margin = self.x_margin.value(), self.y_margin.value()
        except ValueError as e:
            QMessageBox.warning(self, "Export", str(e))
            return

        existing = [out_dir / f"{name}.ibl" for _, name in targets if (out_dir / f"{name}.ibl").exists()]
        if existing and QMessageBox.question(
            self, "Overwrite files?", "Replace\n" + "\n".join(p.name for p in existing) + f"\nin {out_dir}?"
        ) != QMessageBox.Yes:
            return
        if not self.session.try_begin("Exporting to Creo"):
            return

        prop = self.session.prop
        cached = self.export_geom[1] if self.export_geom and self.export_geom[0] is prop else None
        n_camber = self.n_camber.value()

        def work(report):
            r, c, beta, pitch_axis, coords, offsets = cached if cached is not None else geometry.blade_export_geometry(prop)
            written = []
            for kind, name in targets:
                if kind == "parting":
                    written.append(geometry.export_parting_surface_ibl(
                        r, c, beta, coords, np.array(prop.blade.r_foils), name, pitch_axis=pitch_axis,
                        x_margin=x_margin, y_margin=y_margin, n_camber_points=n_camber, out_dir=out_dir,
                        offsets=offsets))
                else:
                    written.append(geometry.export_to_creo_ibl(
                        r, c, beta, coords, name, pitch_axis=pitch_axis,
                        skin_thickness=laminate if kind == "insert" else 0.0, out_dir=out_dir, offsets=offsets))
            return written

        task = Task(work, self)
        task.done.connect(self._exported)
        task.failed.connect(self._task_failed)
        task.finished.connect(self.session.end)
        task.start()

    def _exported(self, paths):
        QMessageBox.information(self, "Creo export", "Written:\n" + "\n".join(str(p) for p in paths))

    def _task_failed(self, text):
        QMessageBox.critical(self, "Failed", text.strip().splitlines()[-1])

    ## Blending
    def blend(self):
        dialog = BlendDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            names, weights = dialog.selection()
            prop = results.propgeomblender(names, weights)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Blend", str(e))
            return
        prop.name = "blend_" + "_".join(names) if len(names) > 1 else names[0]
        self.session.path = None
        self.session.set_propeller(prop)

    def close_views(self):
        if self.interactor is not None:
            self.interactor.close()
