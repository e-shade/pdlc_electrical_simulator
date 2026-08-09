# Modeling and Simulation of Multi-Busbar PDLC Transient and Steady-State Characteristics

## 1. Introduction
Polymer Dispersed Liquid Crystal (PDLC) smart films operate as highly capacitive, large-area electro-optical loads. When an alternating current (AC) is applied across the film's conductive outer layers—typically Indium Tin Oxide (ITO)—an electric field aligns the liquid crystal droplets, transitioning the material from opaque to transparent. 

Because the ITO layers possess finite sheet resistance, a large-area PDLC film acts as a distributed RC network rather than an ideal parallel-plate capacitor. This results in localized voltage drops and non-uniform transition times, heavily dependent on busbar placement. This paper outlines the numerical framework used to simulate the transient and steady-state voltage distributions across coupled ITO layers using a Finite-Difference Time-Domain (FDTD) approach.

---

## 2. Mathematical Model

### 2.1 The Distributed RC Network
A PDLC film consists of two resistive ITO sheets (top and bottom) separated by a dielectric PDLC layer. The electrical behavior of each sheet can be modeled as a two-dimensional diffusion equation governing a distributed RC transmission line. 

Let $V(x,y,t)$ represent the voltage potential at a given point on one of the ITO layers. According to Ohm's Law in two dimensions, the surface current density $\vec{J}$ is directly proportional to the voltage gradient and inversely proportional to the sheet resistance $R_{sq}$ (measured in $\Omega/\text{sq}$):

$$ \vec{J} = -\frac{1}{R_{sq}} \nabla V $$

The divergence of this current density dictates the rate of charge accumulation within the local area capacitor. By applying the continuity equation and substituting the capacitive relationship $q = C_A V$, where $C_A$ is the capacitance per unit area (in $\text{F/m}^2$), we arrive at the governing partial differential equation (PDE) for the diffusion of potential across the film:

$$ C_A \frac{\partial V}{\partial t} = \frac{1}{R_{sq}} \nabla^2 V $$

Rewriting this in terms of the Laplacian operator gives the standard 2D heat/diffusion equation:

$$ \frac{\partial V}{\partial t} = \frac{1}{R_{sq} C_A} \left( \frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} \right) $$

### 2.2 Uncoupled Layer Approximation
In this simulation, the Top and Bottom ITO layers are solved as separate scalar fields, $V_t(x,y,t)$ and $V_b(x,y,t)$. The effective driving voltage across the PDLC core—which determines the optical state—is calculated as the Root Mean Square (RMS) of the differential potential between the two layers:

$$ V_{eff} = \frac{| V_t - V_b |}{\sqrt{2}} $$

This approach computationally decouples the transient step for each layer, allowing them to be solved in parallel before combining them to extract the effective RMS voltage mapping.

---

## 3. Numerical Method (FDTD)

To solve the diffusion equation numerically, we employ an explicit Finite-Difference Time-Domain (FDTD) method using a Forward-Time Central-Space (FTCS) scheme.

### 3.1 Spatial and Temporal Discretization
The film geometry is mapped onto a 2D discrete grid with indices $(i, j)$ corresponding to spatial coordinates, separated by step sizes $\Delta x$ and $\Delta y$. Time is divided into discrete steps of length $\Delta t$.

The continuous Laplacian is approximated using the standard five-point stencil. Assuming an isotropic grid where $\Delta x \approx \Delta y$, the spatial derivative is discretized as:

$$ \nabla^2 V \approx \frac{V_{i+1,j}^n + V_{i-1,j}^n + V_{i,j+1}^n + V_{i,j-1}^n - 4V_{i,j}^n}{\Delta x^2} $$

### 3.2 The Update Equation
Substituting the finite differences into the governing equation yields the explicit time-stepping formula used in the simulator:

$$ V_{i,j}^{n+1} = V_{i,j}^n + \alpha \left( V_{i+1,j}^n + V_{i-1,j}^n + V_{i,j+1}^n + V_{i,j-1}^n - 4V_{i,j}^n \right) $$

where $\alpha$ is a dimensionless diffusion coefficient grouping the physical and numerical parameters:

$$ \alpha = \frac{\Delta t}{R_{sq} C_A \Delta x^2} $$

### 3.3 Stability Criterion
Because the FTCS scheme is conditionally stable for parabolic PDEs, the time step $\Delta t$ must be strictly limited to satisfy the 2D equivalent of the Courant–Friedrichs–Lewy (CFL) condition. To ensure numerical stability without oscillations, the simulator calculates a safe time step bounded by:

$$ \Delta t \le \frac{1}{2} R_{sq} C_A \left( \frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} \right)^{-1} $$

The simulation implements a highly conservative scaling factor of **0.2** on this limit to guarantee stable convergence during long steady-state sweeps.

---

## 4. Boundary Conditions

The simulation relies on a combination of Dirichlet and Neumann boundary conditions to model the physical busbars and the cut edges of the film.

### 4.1 Busbars (Dirichlet Boundaries)
Busbars are defined as geometric masks on the grid where the voltage is held constant over time. Depending on the selected polarity, the fixed nodes are driven at:

$$ V(x,y) = \pm \frac{V_{peak}}{2} $$

During every temporal iteration, grid points overlapping with a busbar coordinate are forcibly overwritten with these target values, acting as infinite-current sources.

### 4.2 Film Edges (Neumann Boundaries)
The physical edges of the ITO film where no busbar exists are electrically insulated, meaning no current can flow out of the film perimeter. This requires a zero-gradient boundary condition orthogonal to the edge normal $\hat{n}$:

$$ \frac{\partial V}{\partial n} = 0 $$

Numerically, this is enforced by copying the voltage values from the adjacent inner cells to the boundary cells at the end of each time step (e.g., $V_{0,j} = V_{1,j}$).

---

## 5. Power Estimation Characteristics

While the transient solver computes field distribution, evaluating the driver requirement necessitates calculating the steady-state AC power. Because PDLC is a highly capacitive load, the dominant current is reactive.

**1. Reactive Current:**
The fundamental capacitive current is dictated by the total film area $A_{total}$, capacitance $C_A$, and the driving frequency $f$ (assumed to be **60 Hz**):

$$ I_{rms} = 2 \pi f (C_A A_{total}) V_{rms} $$

**2. Apparent and Active Power:**
The apparent power (VA) is $S = V_{rms} I_{rms}$. The true active power is estimated by defining an empirical power factor (PF) that scales with both film area and sheet resistance, bounding the PF between **0.05** and **0.85**. 

**3. Busbar ESR Losses:**
Additional active power is dissipated as heat across the copper busbars due to their Equivalent Series Resistance (ESR). The ESR is derived from the busbar's physical dimensions (tape width and length) and its specific sheet resistance $R_{busbar\_sq}$:

$$ P_{ESR} = I_{rms}^2 \sum \left( R_{busbar\_sq} \frac{L_{busbar}}{W_{busbar}} \right) $$

The final estimated DC supply power draw is the sum of the core active power and the busbar ESR thermal losses.