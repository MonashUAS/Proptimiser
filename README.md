# Proptimiser: A JAX-Accelerated Propeller Optimisation Tool

Proptimiser is an advanced, hardware-accelerated tool for the aerodynamic design, evaluation, and optimisation of propellers using Blade Element Momentum Theory (BEMT). It implements state-of-the-art inflow models alongside physics-informed machine learning (NeuralFoil) and analytical aerodynamic coefficients to find the most efficient propeller geometry for any given set of flight conditions.

![alt text](media/proptimiseroutputs.png)

By leveraging [JAX](https://github.com/google/jax) for automatic differentiation and Just-In-Time (JIT) compilation, Proptimiser can efficiently optimise a complex solution space using well-established solvers (like L-BFGS-B and SLSQP) to minimise power consumption while adhering to strict structural and flight constraints.

---

## Features

* **Advanced Aerodynamic Modeling:**
  * Uses a generalised inflow model [3] to evaluate arbitrary forward flight, hover, and off-axis conditions.
  * Captures subsonic drag rises using Karman-Tsien/Prandtl-Glauert (PG) compressibility corrections.
  * Predicts tip and root losses via classical Prandtl loss factors.
* **Physics-Informed Aerodynamics:**
  * Utilises Parametric, XFOIL or **NeuralFoil** [9] for airfoil lookups, providing differentiable aerodynamic coefficients ($C_L$, $C_D$, $C_M$) across operational Reynolds and Mach numbers.
* **Parametric Geometry Generation:**
  * Uses the Kulfan Class Shape Transformation (CST) [13] via **AeroSandbox** [5] to create a fully continuous loft of the blade from the structural hub to the aerodynamic tip.
  * Directly exports production-ready `.ibl` surface meshes and mold-split parting lines directly to **Creo Parametric**.
* **JAX-Accelerated Optimisation:**
  * Multi-objective optimisation (combining Minimum Induced Loss constraints with pure motor power minimisation) using `JAXopt` [8]
  * Optimises blade twist ($\beta$), chord ($c$), and operational RPM ($\omega$) simultaneously.

---

## Installation

Proptimiser requires Python 3.9+ and uses `pip` for dependency management. Since it relies heavily on JAX, it is highly recommended to run this in a virtual environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/MonashUAS/Propcalc-2_prop-er_faster_stronger.git](https://github.com/MonashUAS/Propcalc-2_prop-er_faster_stronger.git)
cd Propcalc-2_prop-er_faster_stronger
```

### 2. Set Up a Virtual Environment (Recommended)

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**On Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required computational and physical libraries:

```bash
pip install -r requirements.txt
```

*(Note: If you plan to use GPU acceleration, ensure you install the CUDA-compatible version of JAX per the [official JAX installation guide](https://www.google.com/search?q=https://jax.readthedocs.io/en/latest/installation.html).)*

---

## CLI & Configuration

Proptimiser is designed to be configured centrally and executed from the command line.

### 1. Configuring Flight Conditions

Before running the solver or evaluating a propeller, open `init.py`. This file contains the `Inputs` dataclass, which controls the bounding conditions of the optimisation.

```python
# init.py
class Inputs(eqx.Module):
    # Solver Configuration
    opt_solver: str = "L-BFGS-B"
    opt_max_iter: int = 1000

    # Geometry Setup
    R_tip: float = 0.25   # Propeller Radius (m)
    R_hub: float = 0.025  # Hub Radius (m)

    # Flight Conditions (Supports Array Batching for Multi-Condition)
    thrust_target = jnp.array([29.6, 22.5]) # Target Thrusts (N)
    V_x = jnp.array([8.60, 0.0])            # Forward Velocity (m/s)
    V_yz = jnp.array([12.29, 0.0])          # Crosswind/Off-axis Velocity (m/s)
    omega = jnp.array([445.0, 400.0])       # Rotor speeds (rad/s)

```

### 2. Running the Optimisation via CLI

Execute the main optimisation loop by running:

```bash
python init.py
```

* **Interactive GUI:** If enabled, a Matplotlib window will spawn, updating the convergence plots, induced velocity fields, and blade shape at every solver step.
* **Terminal Output:** The CLI will log the step time, current Thrust error, and current power consumption.
* **Exporting:** Upon convergence (or hitting the maximum iteration limit), the CLI will prompt: `Save Propeller? (y/n)`. Selecting `y` saves the propeller to a `.eqx` file so flight conditions can be indexed later.

### 3. Results and Post-Processing

* **Multi-Condition Performance Verification:** Feeds saved propeller data back into the propeller modelling data to evaluate a fixed propeller geometry at arbitrary flight conditions.
* **Geometric Smoothing Runs:** Provides a manual override to run control-point smoothing and evaluates effect on blade.

---

```text
--- Evaluating 2 Flight Conditions ---
Resolving Induced Velocity Field for all conditions...
Cond: 0   | Omega: 445.00 | Motor Pwr: 1250.42 | Thrust: 29.61 | MIL RMS: 0.1240
Cond: 1   | Omega: 400.00 | Motor Pwr: 940.15  | Thrust: 22.48 | MIL RMS: 0.0891
```
---

### 4. Geometry Exports

Converts propeller geometry into CAD exports and 3D solid mesh generation.


* **Hub Planform Correction:** Extrapolates aerodynamic data down to the true spin center ($r=0$). This structurally modifies the root chord and twist to prevent center gaps or self-intersections during solid CAD modeling.
* **3D Solid Mesh Generation:** Interpolates continuous Class Shape Transformation (CST) airfoil sections along a 3D quarter-chord pitch axis to construct a polygonal model for rapid local rendering.
* **Tooling & Ply Construction (`.ibl` Exports):** Outputs native Creo-compatible curve tables for three distinct manufacturing targets:
* *Aerodynamic Mold:* The exact external outer mold line (OML) boundary.
* *Internal Insert:* An automatically offset internal core dimension adjusted for composite skin laminate thickness, used for a 3D printed insert.
* *Parting Surface:* The 3D leading and trailing edge spatial trace defining the split line for molding halves.

---

##  References

The underlying aerodynamics and solver mathematics are built upon the following literature:

[1] S. Mancini, K. Boorsma, G. Schepers, and F. Savenije, “A comparison of dynamic inflow models for the blade element momentum method,” *Wind Energ. Sci.*, vol. 8, no. 2, pp. 193–210, Feb. 2023, doi: 10.5194/wes-8-193-2023.

[2] W. Zhong, T. Wang, W. Z. Shen, and W. J. Zhu, “A function improving tip loss correction of blade-element momentum theory for wind turbines,” *International Journal of Sustainable Energy*, vol. 43, no. 1, p. 2352794, Dec. 2024, doi: 10.1080/14786451.2024.2352794.

[3] W. Khan and M. Nahon, “A propeller model for general forward flight conditions,” *International Journal of Intelligent Unmanned Systems*, vol. 3, no. 2/3, pp. 72–92, May 2015, doi: 10.1108/IJIUS-06-2015-0007.

[4] P. Yu, J. Peng, J. Bai, X. Han, and X. Song, “Aeroacoustic and aerodynamic optimization of propeller blades,” *Chinese Journal of Aeronautics*, vol. 33, no. 3, pp. 826–839, Mar. 2020, doi: 10.1016/j.cja.2019.11.005.

[5] P. Sharpe, “AeroSandbox: A Differentiable Framework for Aircraft Design Optimization,” Massachusetts Institute of Technology, 2021. Accessed: Jun. 24, 2026. [Online]. Available: https://hdl.handle.net/1721.1/140023

[6] J. V. R. Prasad, Y.-B. Kong, and D. Peters, “ANALYTICAL METHODS FOR MODELING INFLOW DYNAMICS OF A COAXIAL ROTOR SYSTEM”.

[7] Y. LENG, M. BRONZ, T. JARDIN, and J.-M. MOSCHETTA, “Comparisons of Different Propeller Wake Models for a Propeller-Wing Combination,” p. 15 pages, 2019, doi: 10.13009/EUCASS2019-147.

[8] “JAXopt — JAXopt 0.8 documentation.” Accessed: Mar. 26, 2026. [Online]. Available: https://jaxopt.github.io/stable/

[9] P. Sharpe and R. J. Hansman, “NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning,” Mar. 20, 2025, arXiv: arXiv:2503.16323. doi: 10.48550/arXiv.2503.16323.

[10] “Optax — Optax documentation.” Accessed: Dec. 15, 2025. [Online]. Available: https://optax.readthedocs.io/en/latest/#

[11] D. M. Pitt and D. A. Peters, “THEORETICAL PREDICTION OF DYNAMIC-INFLOW DERIVATIVES.,” *Vertica*, vol. 5, no. 1, pp. 21–34, 1981.

[13] B. M. Kulfan, "Universal Parametric Geometry Representation Method," *Journal of Aircraft*, vol. 45, no. 1, pp. 142–158, 2008, doi: 10.2514/1.29958. *(Used by AeroSandbox for the CST interpolation from the structural hub to aerodynamic stations).*

And more! (will write up into a thesis/paper soon...)