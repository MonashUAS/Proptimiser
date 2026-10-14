## Setup tab: solver settings, flight conditions and blade definition (Inputs + BladeSpec)

# External Imports
import dataclasses
import warnings

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

# Internal Imports
from .. import geometry
from .. import optimiser
from ..init import BladeSpec, Inputs, Propeller, _to_json
from .widgets import ConditionTable, Field, FieldForm, FloatEdit, FreezeTable

MODEL_CHOICES = dict(
    inflow_model=("Simple (uniform)", "Pitt-Peters"),
    coeff_method=("Parametric", "2D interpolation", "3D interpolation", "NeuralFoil"),
    tip_loss_model=("None", "Prandtl"),
    root_loss_model=("None", "Prandtl"),
    rot_corr_model=("None", "Du-Selig"),
    high_induction_model=("Classical momentum", "Buhl"),
)

INPUT_GROUPS = [
    ("Optimiser", [
        Field("opt_solver", "Solver", "text_choice", choices=tuple(optimiser.SOLVERS)),
        Field("opt_max_iter", "Max iterations", "int", minimum=1),
        Field("opt_tol", "Relative tolerance", "float",
              tooltip="Converged when the objective changes by less than this fraction in one iteration"),
    ]),
    ("Induced velocity (BEMT)", [
        Field("V_i_0_max_iter", "Max Newton iterations", "int", minimum=1),
        Field("V_i_0_eps", "Residual tolerance", "float",
              tooltip="Largest non-dimensional BEMT residual accepted (thrust / target, torque / (target x R_tip))"),
        Field("bemt_type", "Solved components", "flags", choices=("Axial", "Tangential", "Radial")),
        Field("n_r", "Radial stations", "int", minimum=3, maximum=500),
        Field("n_psi", "Azimuthal stations", "int", minimum=1, maximum=360),
    ]),
    ("Aerodynamic models", [
        Field("inflow_model", "Inflow", "choice", choices=MODEL_CHOICES["inflow_model"]),
        Field("coeff_method", "Airfoil coefficients", "choice", choices=MODEL_CHOICES["coeff_method"]),
        Field("tip_loss_model", "Tip loss", "choice", choices=MODEL_CHOICES["tip_loss_model"]),
        Field("root_loss_model", "Root loss", "choice", choices=MODEL_CHOICES["root_loss_model"]),
        Field("rot_corr_model", "Rotational correction", "choice", choices=MODEL_CHOICES["rot_corr_model"]),
        Field("high_induction_model", "High induction", "choice", choices=MODEL_CHOICES["high_induction_model"]),
    ]),
    ("Motor", [
        Field("I_0", "No-load current [A]", "float"),
        Field("R_tot", "Motor, ESC and lead resistance [ohm]", "float"),
        Field("kV", "kV [RPM/V]", "float"),
        Field("kappa", "ESC voltage factor", "float", tooltip="1 for a motor on a direct supply"),
        Field("Q0", "No-load friction torque [Nm]", "float"),
        Field("V_bus", "Bus voltage [V]", "float", tooltip="0 sets no limit on the ESC duty"),
    ]),
    ("Constraints", [
        Field("thrust_tol", "Thrust tolerance", "float", tooltip="Relative thrust error allowed about each target"),
        Field("P_max", "Max motor power [W]", "float"),
        Field("min_confidence", "Min NeuralFoil confidence", "float",
              tooltip="Lowest NeuralFoil confidence, averaged over the aerodynamic zone with each element weighted by "
                      "its force scale V_R^2 dr (0 disables)"),
        Field("w_smooth", "Chord smoothness weight", "float",
              tooltip="Weight of the penalty on the second difference of chord over the aerodynamic zone, relative to "
                      "the mean chord (every solver; 0 disables). Stops single stations collapsing to the minimum chord"),
    ]),
    ("Penalty weights (L-BFGS-B only)", [
        Field("w_thrust", "Thrust weight", "float", tooltip="Weight of the thrust and motor power penalties"),
        Field("w_thickness", "Thickness weight", "float", tooltip="Weight of the minimum t/c penalty"),
        Field("w_confidence", "Confidence weight", "float", tooltip="Weight of the NeuralFoil low-confidence penalty"),
    ]),
    ("Diagnostics", [
        Field("opt_verbose", "Optimiser verbosity", "int", maximum=2,
              tooltip="0: start/end summary only, 1: iteration table, 2: + constraint margins and solver detail"),
        Field("opt_print_every", "Report every (iterations)", "int", minimum=1),
        Field("V_i_0_verbose", "BEMT verbosity", "int", maximum=2,
              tooltip="0: silent, 1: one line per induced velocity solve, 2: + every Newton iteration"),
        Field("jax_tracing", "JAX profiler trace", "bool"),
    ]),
]

BLADE_GROUPS = [
    ("Blade definition", [
        Field("N", "Blades", "int", minimum=1, maximum=12),
        Field("r_foils", "r_foils [m]", "floats",
              tooltip="0, r_hub, r_aero, r_mid..., r_tip: hub / blend / aerodynamic zones; CST sections sit at r_foils[2:]"),
        Field("airfoils", "Seed airfoils", "strings",
              tooltip="One name for every aerodynamic station, or one per station in r_foils[2:]"),
        Field("hub_chord", "Hub chord [m]", "float"),
        Field("hub_twist_deg", "Hub twist [deg]", "float"),
        Field("hub_thickness_ratio", "Hub t/c", "float"),
    ]),
    ("Design variables", [
        Field("optimise_cst", "Optimise airfoil sections", "bool", tooltip="Requires NeuralFoil coefficients"),
        Field("c_bounds", "Chord bounds [m]", "pair"),
        Field("beta_bounds_deg", "Twist bounds [deg]", "pair"),
        Field("cst_bound", "CST weight bound (+/-)", "float"),
        Field("min_tc", "Min t/c", "float"),
    ]),
    ("Sweep and rake", [
        Field("offset_r", "Control radii [m]", "floats",
              tooltip="Radii where the pitch-axis offsets below are given, strictly increasing. Empty: an unswept blade"),
        Field("offset_y", "Sweep [m]", "floats",
              tooltip="Tangential offset of the pitch axis, positive toward the direction of rotation (APC SWEEP): "
                      "a backswept tip is negative. One value per control radius, or empty for none"),
        Field("offset_z", "Rake [m]", "floats",
              tooltip="Axial offset of the pitch axis, positive toward the thrust direction (APC RAKE). "
                      "One value per control radius, or empty for none"),
    ]),
    ("Structure", [
        Field("mat_density", "Density [kg/m^3]", "float"),
        Field("mat_E", "Young's modulus E [Pa]", "float"),
        Field("mat_G", "Shear modulus G [Pa]", "float"),
        Field("mat_sigma_allow", "Allowable normal stress [Pa]", "float"),
        Field("mat_tau_allow", "Allowable shear stress [Pa]", "float", tooltip="0: allowable normal stress / sqrt(3)"),
        Field("fill_factor", "Fill factor", "float",
              tooltip="Share of the section outline that is material: 1 is solid, below 1 is a homogenised infill"),
    ]),
]


def inputs_dict(inputs) -> dict:
    return _to_json({f.name: getattr(inputs, f.name) for f in dataclasses.fields(inputs)})


class SetupTab(QWidget):
    '''
    Edits the run configuration of the current propeller. Changes take effect on Apply, which rebuilds the
    propeller with Propeller.with_inputs / with_blade (re-gridding chord and twist when n_r changes).
    '''
    pending_changed = Signal(bool)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._pending = False

        # Forms
        self.conditions = ConditionTable()
        self.input_forms = [(title, FieldForm(fields)) for title, fields in INPUT_GROUPS]
        self.blade_forms = [(title, FieldForm(fields)) for title, fields in BLADE_GROUPS]
        self.freeze = FreezeTable()

        for editor in [self.conditions, self.freeze] + [form for _, form in self.input_forms + self.blade_forms]:
            editor.edited.connect(self._mark_pending)

        (blade_title, blade_form), (design_title, design_form), *extra_blade_forms = self.blade_forms
        design_box = self._group(design_title, design_form)
        design_box.layout().addWidget(QLabel("Frozen radial ranges (held at their seed values)"))
        design_box.layout().addWidget(self.freeze)

        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(self._group("Flight conditions", self.conditions))
        for title, form in self.input_forms[:3]:
            left.addWidget(self._group(title, form))
        right.addWidget(self._group(blade_title, blade_form))
        right.addWidget(design_box)
        for title, form in extra_blade_forms:
            right.addWidget(self._group(title, form))
        for title, form in self.input_forms[3:]:
            right.addWidget(self._group(title, form))
        left.addStretch()
        right.addStretch()

        columns = QWidget()
        grid = QGridLayout(columns)
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(columns)

        # Action bar
        self.apply_button = QPushButton("Apply")
        self.revert_button = QPushButton("Revert")
        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        self.status = QLabel()
        self.status.setWordWrap(True)

        self.seed_chord = FloatEdit(20.0)
        self.seed_twist = FloatEdit(10.0)
        for edit in (self.seed_chord, self.seed_twist):
            edit.setMaximumWidth(70)
        new_button = QPushButton("New blade from settings")
        new_button.setToolTip("Constant-chord, constant-twist blade built from the settings above")
        new_button.clicked.connect(self.new_from_settings)

        actions = QHBoxLayout()
        actions.addWidget(self.apply_button)
        actions.addWidget(self.revert_button)
        actions.addWidget(self.status, 1)
        actions.addWidget(QLabel("Seed chord [mm]"))
        actions.addWidget(self.seed_chord)
        actions.addWidget(QLabel("twist [deg]"))
        actions.addWidget(self.seed_twist)
        actions.addWidget(new_button)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(actions)

        session.propeller_changed.connect(self.load)
        self.load(session.prop)

    @staticmethod
    def _group(title, widget):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    ## State
    def has_pending_changes(self) -> bool:
        return self._pending

    def _set_pending(self, pending, message=""):
        self._pending = pending
        self.apply_button.setEnabled(pending)
        self.revert_button.setEnabled(pending)
        self.status.setStyleSheet("")
        self.status.setText(message or ("Unapplied changes" if pending else ""))
        self.pending_changed.emit(pending)

    def _mark_pending(self):
        if not self._pending:
            self._set_pending(True)

    def _error(self, message):
        self.status.setStyleSheet("color: #c0392b;")
        self.status.setText(message)

    def load(self, prop):
        '''
        Fills every form from prop, discarding unapplied edits.
        '''
        self.conditions.set_inputs(prop.inputs)
        for _, form in self.input_forms:
            form.set_values(prop.inputs)
        for _, form in self.blade_forms:
            form.set_values(prop.blade)
        self.freeze.set_freeze(prop.blade.freeze)
        self._set_pending(False)

    def revert(self):
        self.load(self.session.prop)

    ## Building configurations
    def _build(self, prop):
        '''
        Inputs and BladeSpec from the forms (raises ValueError with a readable message on bad input).
        '''
        values = {}
        for _, form in self.input_forms:
            values.update(form.values())
        values.update(self.conditions.values())
        current = {f.name: getattr(prop.inputs, f.name) for f in dataclasses.fields(prop.inputs)}
        inputs = Inputs(**{**current, **values})

        blade_values = {}
        for _, form in self.blade_forms:
            blade_values.update(form.values())
        blade_values["freeze"] = self.freeze.values()

        old = prop.blade
        base = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)}
        if blade_values["hub_thickness_ratio"] != old.hub_thickness_ratio:
            base["hub_cst"] = None      # refit the hub section to the new thickness
        blade = BladeSpec(**{**base, **blade_values})
        return inputs, blade

    def apply(self):
        prop = self.session.prop
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                inputs, blade = self._build(prop)
                new = prop
                if inputs_dict(inputs) != inputs_dict(prop.inputs):
                    new = new.with_inputs(inputs)
                if blade != prop.blade:
                    new = new.with_blade(blade)
                    new = self._reseed_changed_airfoils(new, prop.blade, blade)
        except (ValueError, TypeError, FileNotFoundError) as e:
            self._error(f"Not applied: {e}")
            return

        if new is prop:
            self._set_pending(False, "No changes")
            return
        self.session.set_propeller(new)
        note = "; ".join(str(w.message) for w in caught)
        self._set_pending(False, f"Applied. {note}" if note else "Applied")

    @staticmethod
    def _reseed_changed_airfoils(prop, old_blade, blade):
        # with_blade keeps the CST sections when the station count is unchanged, so a renamed seed airfoil
        # re-seeds only its own station
        if blade.n_aero != old_blade.n_aero:
            return prop
        changed = [j for j, (a, b) in enumerate(zip(old_blade.airfoils, blade.airfoils)) if a != b]
        if not changed:
            return prop
        cst = np.array(prop.cst)
        cst[changed] = geometry.get_cst_weights_from_names([blade.airfoils[j] for j in changed])
        return prop.with_state(cst=cst)

    def new_from_settings(self):
        if self.session.dirty and QMessageBox.question(
            self, "New blade", "Replace the current design with a new constant-chord blade? Unsaved changes are lost."
        ) != QMessageBox.Yes:
            return
        try:
            inputs, blade = self._build(self.session.prop)
            blade = dataclasses.replace(blade, hub_cst=None)
            prop = Propeller.default(inputs=inputs, blade=blade, c=self.seed_chord.value() / 1000.0,
                                     beta_deg=self.seed_twist.value())
        except (ValueError, TypeError, FileNotFoundError) as e:
            self._error(f"Not created: {e}")
            return
        self.session.path = None
        self.session.set_propeller(prop)
        self._set_pending(False, "New blade created")
