import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDLC Distributed Transient Simulator", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { min-width: 400px !important; }
    .katex-display { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Interactive PDLC Voltage Distribution Simulator")
st.caption("Finite-Difference Time-Domain (FDTD) solver for large-format capacitive smart glass loads.")

# --- SIDEBAR GUI ---
with st.sidebar:
    st.header("Film Geometry")
    film_w = st.slider("Width (m)", 0.1, 3.0, 1.0, step=0.1)
    film_l = st.slider("Length (m)", 0.1, 3.0, 1.0, step=0.1)

    st.header("Electrical Properties")
    r_sq = st.number_input("ITO Sheet Resistance (Ω/sq)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, value=11.0, step=0.5)
    c_area = c_area_uf * 1e-6

    st.header("Busbar Configuration")
    st.write("Select edges to apply copper busbars:")
    bb_left = st.checkbox("Left Edge", value=True)
    bb_right = st.checkbox("Right Edge", value=True)
    bb_top = st.checkbox("Top Edge", value=False)
    bb_bottom = st.checkbox("Bottom Edge", value=False)

    st.header("Simulation Settings")
    v_drive = st.slider("Driving Voltage (V)", 10.0, 150.0, 60.0, step=5.0)
    # Use microseconds for a cleaner UI
    t_snapshot_us = st.slider("Snapshot Time (µs)", 10, 5000, 500, step=10)
    t_snapshot = t_snapshot_us * 1e-6
    resolution = st.select_slider("Grid Resolution", options=[20, 40, 60, 80], value=40)

# --- FDTD SOLVER (Cached for performance) ---
@st.cache_data
def simulate_transients(C_A, R_sq, width, length, bb_config, V_drive, t_snap, nx, ny):
    dx = width / (nx - 1)
    dy = length / (ny - 1)
    
    # Courant–Friedrichs–Lewy stability condition
    dt = 0.2 * (R_sq * C_A) / (1/(dx**2) + 1/(dy**2))
    steps = int(t_snap / dt)
    
    alpha_x = dt / (R_sq * C_A * dx**2)
    alpha_y = dt / (R_sq * C_A * dy**2)

    def apply_busbars(V, target_V):
        if bb_config['left']:   V[:, 0] = target_V
        if bb_config['right']:  V[:, -1] = target_V
        if bb_config['top']:    V[0, :] = target_V
        if bb_config['bottom']: V[-1, :] = target_V
        return V

    def solve(initial_V, target_V):
        V = initial_V.copy()
        V = apply_busbars(V, target_V)
        
        for _ in range(steps):
            V_new = V.copy()
            V_new[1:-1, 1:-1] = V[1:-1, 1:-1] + \
                                alpha_x * (V[1:-1, 2:] - 2*V[1:-1, 1:-1] + V[1:-1, :-2]) + \
                                alpha_y * (V[2:, 1:-1] - 2*V[1:-1, 1:-1] + V[:-2, 1:-1])
            
            # Neumann boundary conditions (No current out of cut edges)
            if not bb_config['top']:    V_new[0, 1:-1] = V_new[1, 1:-1]
            if not bb_config['bottom']: V_new[-1, 1:-1] = V_new[-2, 1:-1]
            if not bb_config['left']:   V_new[:, 0] = V_new[:, 1]
            if not bb_config['right']:  V_new[:, -1] = V_new[:, -2]
            
            V_new = apply_busbars(V_new, target_V)
            V = V_new
        return V

    V_on = solve(np.zeros((ny, nx)), V_drive)
    V_off = solve(np.full((ny, nx), V_drive), 0.0)
    
    return V_on, V_off, dx, dy, steps

# --- EXECUTE SIMULATION ---
with st.spinner("Calculating RC Diffusion..."):
    bb_dict = {'left': bb_left, 'right': bb_right, 'top': bb_top, 'bottom': bb_bottom}
    V_on_res, V_off_res, dx_val, dy_val, total_steps = simulate_transients(
        c_area, r_sq, film_w, film_l, bb_dict, v_drive, t_snapshot, resolution, resolution
    )

# --- PLOTTING ---
x_grid = np.linspace(0, film_w, resolution)
y_grid = np.linspace(0, film_l, resolution)

def create_3d_surface(Z, title, colorscale):
    fig = go.Figure(data=[go.Surface(z=Z, x=x_grid, y=y_grid, colorscale=colorscale)])
    fig.update_layout(
        title=title,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='Width (m)',
            yaxis_title='Length (m)',
            zaxis_title='Voltage (V)',
            zaxis=dict(range=[0, v_drive])
        )
    )
    return fig

col1, col2 = st.columns(2)

with col1:
    fig_on = create_3d_surface(V_on_res, f"Turn-ON (t = {t_snapshot_us} µs)", "Inferno")
    st.plotly_chart(fig_on, use_container_width=True)

with col2:
    fig_off = create_3d_surface(V_off_res, f"Turn-OFF (t = {t_snapshot_us} µs)", "Viridis")
    st.plotly_chart(fig_off, use_container_width=True)

# --- METRICS BANNERS ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total RC Area", f"{film_w * film_l:.2f} m²")
m2.metric("Center Node Turn-ON Voltage", f"{V_on_res[resolution//2, resolution//2]:.2f} V")
m3.metric("Center Node Turn-OFF Voltage", f"{V_off_res[resolution//2, resolution//2]:.2f} V")
m4.metric("FDTD Calculation Steps", f"{total_steps:,}")