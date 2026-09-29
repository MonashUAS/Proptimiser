## Reusable widgets: embedded matplotlib figures, declarative forms and editable tables

# External Imports
import math
import warnings
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLineEdit, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# Internal Imports
import results
from init import CONDITION_FIELDS


class MplCanvas(QWidget):
    '''
    A matplotlib Figure embedded in Qt, with the standard navigation toolbar (zoom, pan, save).
    '''
    def __init__(self, parent=None, figsize=(10, 6), toolbar=True):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

        # Figures are built before the widget has its final size, so re-fit the layout whenever it changes
        self.canvas.mpl_connect("resize_event", lambda event: self.relayout())

    def relayout(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # e.g. colorbar axes that tight_layout cannot place
            try:
                self.figure.tight_layout()
            except (ValueError, RuntimeError):
                pass

    def message(self, text):
        '''
        Clears the figure and shows a centred message (e.g. "Evaluate the design to see results").
        '''
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="0.4", wrap=True)
        self.canvas.draw_idle()

    def draw(self):
        self.relayout()
        self.canvas.draw_idle()


class LivePlotCanvas(MplCanvas):
    '''
    MplCanvas showing the results.LivePropPlotter panels for FlowSnapshots.
    The plotter's lines are sized to the grid, so it is rebuilt whenever the grid changes.
    '''
    def __init__(self, parent=None):
        super().__init__(parent, figsize=(18, 9))
        self.plotter = None
        self._key = None

    def show_snapshot(self, snap):
        key = (len(snap.r), len(snap.psi), len(snap.r_cst))
        if self.plotter is None or key != self._key:
            self.figure.clear()
            self.plotter = results.LivePropPlotter(snap, fig=self.figure)
            self._key = key
        self.plotter.draw(snap)
        self.relayout()

    def message(self, text):
        self.plotter = None
        super().message(text)


class FloatEdit(QLineEdit):
    '''
    Line edit for a float, accepting scientific notation (e.g. 1e-4).
    '''
    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setValidator(validator)
        self.setValue(value)

    def value(self) -> float:
        text = self.text().strip()
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a number") from None

    def setValue(self, value):
        self.setText(f"{float(value):.10g}")


def _parse_list(text, cast):
    items = [s.strip() for s in text.replace(";", ",").split(",") if s.strip()]
    try:
        return tuple(cast(s) for s in items)
    except ValueError:
        raise ValueError(f"'{text}' is not a comma-separated list") from None


@dataclass
class Field:
    '''
    One form row bound to a dataclass attribute.

    kind: "int", "float", "bool", "choice" (index into choices), "text", "text_choice" (one of choices, as text),
          "floats" / "strings" (comma-separated tuple), "pair" (two floats), "flags" (tuple of bools labelled by choices)
    '''
    name: str
    label: str
    kind: str
    choices: tuple = ()
    minimum: int = 0
    maximum: int = 1_000_000
    tooltip: str = ""


class FieldForm(QWidget):
    '''
    Form built from a list of Fields. values() returns {name: value}; `edited` fires on any user change.
    '''
    edited = Signal()

    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.fields = list(fields)
        self.editors = {}
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        for f in self.fields:
            editor = self._make_editor(f)
            if f.tooltip:
                editor.setToolTip(f.tooltip)
            self.editors[f.name] = editor
            layout.addRow(f.label, editor)

    def _make_editor(self, f: Field):
        if f.kind == "int":
            editor = QSpinBox()
            editor.setRange(f.minimum, f.maximum)
            editor.valueChanged.connect(self.edited)
        elif f.kind == "float":
            editor = FloatEdit()
            editor.textEdited.connect(self.edited)
        elif f.kind == "bool":
            editor = QCheckBox()
            editor.toggled.connect(self.edited)
        elif f.kind in ("choice", "text_choice"):
            editor = QComboBox()
            editor.addItems(list(f.choices))
            editor.currentIndexChanged.connect(self.edited)
        elif f.kind in ("text", "floats", "strings"):
            editor = QLineEdit()
            editor.textEdited.connect(self.edited)
        elif f.kind == "pair":
            editor = QWidget()
            row = QHBoxLayout(editor)
            row.setContentsMargins(0, 0, 0, 0)
            editor.parts = (FloatEdit(), FloatEdit())
            for part in editor.parts:
                part.textEdited.connect(self.edited)
                row.addWidget(part)
        elif f.kind == "flags":
            editor = QWidget()
            row = QHBoxLayout(editor)
            row.setContentsMargins(0, 0, 0, 0)
            editor.parts = tuple(QCheckBox(label) for label in f.choices)
            for part in editor.parts:
                part.toggled.connect(self.edited)
                row.addWidget(part)
            row.addStretch()
        else:
            raise ValueError(f"Unknown field kind {f.kind}")
        return editor

    def set_values(self, obj):
        '''
        Loads every field from the attributes of obj without emitting `edited`.
        '''
        for f in self.fields:
            editor, value = self.editors[f.name], getattr(obj, f.name)
            blocked = [editor] + list(getattr(editor, "parts", ()))
            for w in blocked:
                w.blockSignals(True)
            try:
                self._set(f, editor, value)
            finally:
                for w in blocked:
                    w.blockSignals(False)

    @staticmethod
    def _set(f, editor, value):
        if f.kind == "int":
            editor.setValue(int(value))
        elif f.kind == "float":
            editor.setValue(value)
        elif f.kind == "bool":
            editor.setChecked(bool(value))
        elif f.kind == "choice":
            editor.setCurrentIndex(int(value))
        elif f.kind == "text_choice":
            editor.setCurrentText(str(value))
        elif f.kind == "text":
            editor.setText(str(value))
        elif f.kind == "floats":
            editor.setText(", ".join(f"{float(v):.10g}" for v in value))
        elif f.kind == "strings":
            editor.setText(", ".join(str(v) for v in value))
        elif f.kind == "pair":
            for part, v in zip(editor.parts, value):
                part.setValue(v)
        elif f.kind == "flags":
            for part, v in zip(editor.parts, value):
                part.setChecked(bool(v))

    def values(self) -> dict:
        '''
        {name: value} for every field; raises ValueError naming the field on unparseable input.
        '''
        out = {}
        for f in self.fields:
            editor = self.editors[f.name]
            try:
                out[f.name] = self._get(f, editor)
            except ValueError as e:
                raise ValueError(f"{f.label}: {e}") from None
        return out

    @staticmethod
    def _get(f, editor):
        if f.kind == "int":
            return editor.value()
        if f.kind == "float":
            return editor.value()
        if f.kind == "bool":
            return editor.isChecked()
        if f.kind == "choice":
            return editor.currentIndex()
        if f.kind == "text_choice":
            return editor.currentText()
        if f.kind == "text":
            return editor.text().strip()
        if f.kind == "floats":
            return _parse_list(editor.text(), float)
        if f.kind == "strings":
            return _parse_list(editor.text(), str)
        if f.kind == "pair":
            return tuple(part.value() for part in editor.parts)
        if f.kind == "flags":
            return tuple(part.isChecked() for part in editor.parts)


def make_table(headers, stretch=True):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    if stretch:
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return table


def fit_table_height(table):
    '''
    Fixes a table's height to show its header and every row without scrolling.
    '''
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    height += sum(table.rowHeight(i) for i in range(table.rowCount()))
    if table.horizontalScrollBar().isVisible():
        height += table.horizontalScrollBar().height()
    table.setFixedHeight(height)


def fill_table(table, rows, formats=None, editable=False):
    '''
    Fills a QTableWidget with rows of values; formats[j] formats column j (default "{:.4g}").
    '''
    table.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            fmt = formats[j] if formats else "{:.4g}"
            text = value if isinstance(value, str) else fmt.format(value)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if not editable:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, j, item)


class ConditionTable(QWidget):
    '''
    Editable flight conditions: one row per condition, one column per Inputs condition field.
    '''
    edited = Signal()

    COLUMNS = {
        "thrust_target": "Thrust target [N]",
        "V_x": "V_x [m/s]",
        "V_yz": "V_yz [m/s]",
        "omega": "Omega [rad/s]",
        "tilt": "Tilt [rad]",
        "solve_omega": "Solve omega",
        "solve_tilt": "Solve tilt",
        "objective_weight": "Weight",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        assert tuple(self.COLUMNS) == CONDITION_FIELDS
        self.table = make_table(list(self.COLUMNS.values()))
        self.table.itemChanged.connect(self._changed)

        add, remove = QPushButton("Add condition"), QPushButton("Remove selected")
        add.clicked.connect(self.add_row)
        remove.clicked.connect(self.remove_rows)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def _changed(self, *_):
        self.edited.emit()

    def _set_row(self, i, values):
        for j, name in enumerate(self.COLUMNS):
            item = QTableWidgetItem()
            if name.startswith("solve_"):
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setCheckState(Qt.Checked if values[name] else Qt.Unchecked)
            else:
                item.setText(f"{float(values[name]):.10g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, j, item)

    def set_inputs(self, inputs):
        self.table.blockSignals(True)
        self.table.setRowCount(inputs.n_cond)
        for i in range(inputs.n_cond):
            self._set_row(i, {name: np.asarray(getattr(inputs, name))[i] for name in self.COLUMNS})
        self.table.blockSignals(False)

    def add_row(self):
        # New conditions copy the last one (or sensible defaults for an empty table)
        n = self.table.rowCount()
        if n:
            values = self.values()
            values = {k: v[-1] for k, v in values.items()}
        else:
            values = dict(thrust_target=10.0, V_x=0.0, V_yz=0.0, omega=500.0, tilt=0.0,
                          solve_omega=False, solve_tilt=False, objective_weight=1.0)
        self.table.blockSignals(True)
        self.table.insertRow(n)
        self._set_row(n, values)
        self.table.blockSignals(False)
        self.edited.emit()

    def remove_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if len(rows) >= self.table.rowCount():
            rows = rows[:-1]        # keep at least one condition
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self.edited.emit()

    def values(self) -> dict:
        '''
        {field: list of per-condition values}; raises ValueError on unparseable cells.
        '''
        out = {name: [] for name in self.COLUMNS}
        for i in range(self.table.rowCount()):
            for j, name in enumerate(self.COLUMNS):
                item = self.table.item(i, j)
                if name.startswith("solve_"):
                    out[name].append(item.checkState() == Qt.Checked)
                    continue
                try:
                    value = float(item.text())
                except (AttributeError, ValueError):
                    raise ValueError(f"Condition {i}: '{self.COLUMNS[name]}' must be a number") from None
                if not math.isfinite(value):
                    raise ValueError(f"Condition {i}: '{self.COLUMNS[name]}' must be finite")
                out[name].append(value)
        return out


class FreezeTable(QWidget):
    '''
    Editable BladeSpec.freeze: rows of (variable, r_lo, r_hi) radial ranges held at their seed values.
    '''
    edited = Signal()
    VARIABLES = ("c", "beta", "cst")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = make_table(["Variable", "r_lo [m]", "r_hi [m]"])
        self.table.setMinimumHeight(90)
        self.table.itemChanged.connect(self._changed)

        add, remove = QPushButton("Add range"), QPushButton("Remove selected")
        add.clicked.connect(lambda: self._add(("c", 0.0, 0.0)))
        remove.clicked.connect(self.remove_rows)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def _changed(self, *_):
        self.edited.emit()

    def _add(self, entry, emit=True):
        var, lo, hi = entry
        i = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(i)
        combo = QComboBox()
        combo.addItems(self.VARIABLES)
        combo.setCurrentText(var)
        combo.currentIndexChanged.connect(self._changed)
        self.table.setCellWidget(i, 0, combo)
        for j, v in ((1, lo), (2, hi)):
            self.table.setItem(i, j, QTableWidgetItem(f"{float(v):.10g}"))
        self.table.blockSignals(False)
        if emit:
            self.edited.emit()

    def remove_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self.edited.emit()

    def set_freeze(self, freeze: dict):
        self.table.setRowCount(0)
        for var, ranges in freeze.items():
            for lo, hi in ranges:
                self._add((var, lo, hi), emit=False)

    def values(self) -> dict:
        out = {}
        for i in range(self.table.rowCount()):
            var = self.table.cellWidget(i, 0).currentText()
            try:
                lo, hi = (float(self.table.item(i, j).text()) for j in (1, 2))
            except (AttributeError, ValueError):
                raise ValueError(f"Freeze range {i}: r_lo and r_hi must be numbers") from None
            out.setdefault(var, []).append((lo, hi))
        return out
