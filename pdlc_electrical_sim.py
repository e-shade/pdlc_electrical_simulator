import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Multi-Busbar PDLC Simulator", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { min-width: 450px !important; }
    .katex-display { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Advanced Multi-Busbar PDLC Transient Simulator")
st.caption("Coupled top/bottom ITO layer finite-difference solver: Turn-ON (Left) vs. Turn-OFF (Right) Transients.")

# --- SIDEBAR GUI ---
with st.sidebar:
    st.header("1. Film Geometry")
    film_w = st.slider("Width (m)", 0.1, 3.0, 1.0, step=0.1)
    film_l = st.slider("Length (m)", 0.1, 3.0, 1.0, step=0.1)

    st.header("2. Electrical Properties")
    r_sq = st.number_input("ITO Sheet Resistance (Ω/sq)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, value=11.0, step=0.5)
    c_area = c_area_uf * 1e-6
    v_drive = st.slider("Driving Voltage Magnitude (V)", 10.0, 150.0, 60.0, step=5.0)

    st.header("3. Multi-Busbar Configuration")
    if 'busbars' not in st.session_state:
        st.session_state.busbars = [
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.0, 'y_end': 0.4, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.6, 'y_end': 1.0, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
        ]

    col_add, col_rem = st.columns(2)
    if col_add.button("➕ Add Busbar"):
        st.session_state.busbars.append(
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.0, 'y_end': 0.5, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
        )
    if col_rem.button("🗑️ Remove Last") and len(st.session_state.busbars) > 1:
        st.session_state.busbars.pop()

    configured_busbars = []
    for i, bb in enumerate(st.session_state.busbars):
        with st.expander(f"Busbar #{i+1} ({bb['layer']} - {bb['signal']})", expanded=True):
            col_x1, col_x2 = st.columns(2)
            x_start = col_x1.number_input(f"X start (m) #{i+1}", 0.0, film_w, float(bb['x_start']), step=0.05, key=f"x1_{i}")
            x_end = col_x2.number_input(f"X end (m) #{i+1}", 0.0, film_w, float(bb['x_end']), step=0.05, key=f"x2_{i}")
            
            col_y1, col_y2 = st.columns(2)
            y_start = col_y1.number_input(f"Y start (m) #{i+1}", 0.0, film_l, float(bb['y_start']), step=0.05, key=f"y1_{i}")
            y_end = col_y2.number_input(f"Y end (m) #{i+1}", 0.0, film_l, float(bb['y_end']), step=0.05, key=f"y2_{i}")
            
            col_l, col_s = st.columns(2)
            layer = col_l.selectbox(f"Layer #{i+1}", ["Top ITO", "Bottom ITO"], index=0 if bb['layer']=='Top ITO' else 1, key=f"lay_{i}")
            signal = col_s.selectbox(f"Signal #{i+1}", ["Positive (+)", "Negative (-)"], index=0 if bb['signal']=='Positive (+)' else 1, key=f"sig_{i}")
            
            configured_busbars.append({
                'x_min': min(x_start, x_end), 'x_max': max(x_start, x_end),
                'y_min': min(y_start, y_end), 'y_max': max(y_start, y_end),
                'layer': layer, 'signal': signal
            })

    st.header("4. Simulation Settings")
    t_snapshot_us = st.slider("Snapshot Time (µs)", 10, 5000, 500, step=10)
    t_snapshot = t_snapshot_us * 1e-6
    resolution = st.select_slider("Grid Resolution", options=[20, 40, 60], value=40)

# --- COUPLED FDTD SOLVER FOR TURN-ON & TURN-OFF ---
@st.cache_data
def simulate_on_off_transients(C_A, R_sq, width, length, busbars, V_mag, t_snap, nx, ny):
    dx = width / (nx - 1)
    dy = length / (ny - 1)
    
    dt = 0.2 * (R_sq * C_A) / (1/(dx**2) + 1/(dy**2))
    steps = int(t_snap / dt)
    alpha = dt / (R_sq * C_A * dx**2)

    def get_busbar_masks_and_targets(V_top, V_bot):
        top_pos_mask, top_val_mask = np.zeros((ny, nx), dtype=bool), np.zeros((ny, nx))
        bot_pos_mask, bot_val_mask = np.zeros((ny, nx), dtype=bool), np.zeros((ny, nx))
        
        has_top, has_bot = False, False
        for bb in busbars:
            ix_min = max(0, int(bb['x_min'] / width * (nx - 1)))
            ix_max = min(nx - 1, int(bb['x_max'] / width * (nx - 1)))
            iy_min = max(0, int(bb['y_min'] / length * (ny - 1)))
            iy_max = min(ny - 1, int(bb['y_max'] / length * (ny - 1)))
            
            val = V_mag if bb['signal'] == 'Positive (+)' else -V_mag
            if bb['layer'] == 'Top ITO':
                top_pos_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                top_val_mask[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_top = True
            else:
                bot_pos_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                bot_val_mask[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_bot = True
                
        if not has_top: V_top[:, :] = 0.0
        if not has_bot: V_bot[:, :] = 0.0
        return top_pos_mask, top_val_mask, bot_pos_mask, bot_val_mask

    # Helper solver execution
    def run_fdtd(init_top, init_bot, target_top_val, target_bot_val, top_mask, bot_mask):
        V_t, V_b = init_top.copy(), init_bot.copy()
        V_t[top_mask] = target_top_val[top_mask]
        V_b[bot_mask] = target_bot_val[bot_mask]
        
        for _ in range(steps):
            V_t_new = V_t.copy()
            V_t_new[1:-1, 1:-1] = V_t[1:-1, 1:-1] + alpha * (
                V_t[1:-1, 2:] + V_t[1:-1, :-2] + V_t[2:, 1:-1] + V_t[:-2, 1:-1] - 4*V_t[1:-1, 1:-1]
            )
            V_b_new = V_b.copy()
            V_b_new[1:-1, 1:-1] = V_b[1:-1, 1:-1] + alpha * (
                V_b[1:-1, 2:] + V_b[1:-1, :-2] + V_b[2:, 1:-1] + V_b[:-2, 1:-1] - 4*V_b[1:-1, 1:-1]
            )
            
            # Enforce constraints
            V_t_new[top_mask] = target_top_val[top_mask]
            V_b_new[bot_mask] = target_bot_val[bot_mask]
            
            # Neumann boundaries
            for V_arr in [V_t_new, V_b_new]:
                V_arr[0, :] = V_arr[1, :]
                V_arr[-1, :] = V_arr[-2, :]
                V_arr[:, 0] = V_arr[:, 1]
                V_arr[:, -1] = V_arr[:, -2]
                
            V_t_new[top_mask] = target_top_val[top_mask]
            V_b_new[bot_mask] = target_bot_val[bot_mask]
            
            V_t, V_b = V_t_new, V_b_new
        return V_t, V_b

    # Masks
    dummy_t, dummy_b = np.zeros((ny, nx)), np.zeros((ny, nx))
    t_mask, t_vals, b_mask, b_vals = get_busbar_masks_and_targets(dummy_t, dummy_b)

    # 1. TURN-ON: Starts from 0V, charges up toward busbar targets
    init_on_t, init_on_b = np.zeros((ny, nx)), np.zeros((ny, nx))
    final_t_on, final_b_on = run_fdtd(init_on_t, init_on_b, t_vals, b_vals, t_mask, b_mask)
    V_eff_on = np.abs(final_t_on - final_b_on)

    # 2. TURN-OFF: Starts from steady-state charged condition, discharges toward 0V
    steady_t, steady_b = t_vals.copy(), b_vals.copy() # Simplified steady-state approximation
    steady_t[~t_mask] = 0.0 # Unconstrained regions start fully charged or interpolated
    steady_b[~b_mask] = 0.0
    
    # Discharge target is 0V everywhere
    zero_vals = np.zeros((ny, nx))
    final_t_off, final_b_off = run_fdtd(steady_t, steady_b, zero_vals, zero_vals, t_mask, b_mask)
    V_eff_off = np.abs(final_t_off - final_b_off)

    return V_eff_on, V_eff_off, steps

# --- EXECUTE ---
with st.spinner("Calculating Turn-ON and Turn-OFF differential profiles..."):
    V_on, V_off, total_steps = simulate_on_off_transients(
        c_area, r_sq, film_w, film_l, configured_busbars, v_drive, t_snapshot, resolution, resolution
    )

# --- PLOTTING WITH MARKERS ---
x_grid = np.linspace(0, film_w, resolution)
y_grid = np.linspace(0, film_l, resolution)

def add_busbar_traces(fig):
    for i, bb in enumerate(configured_busbars):
        bx = [bb['x_min'], bb['x_max'], bb['x_max'], bb['x_min'], bb['x_min']]
        by = [bb['y_min'], bb['y_min'], bb['y_max'], bb['y_max'], bb['y_min']]
        bz = [v_drive * 2.05] * 5
        color = "red" if bb['signal'] == 'Positive (+)' else "blue"
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color=color, width=6),
            name=f"BB#{i+1} ({bb['layer']} {bb['signal']})"
        ))

col1, col2 = st.columns(2)

with col1:
    fig_on = go.Figure(data=[go.Surface(z=V_on, x=x_grid, y=y_grid, colorscale="Inferno")])
    add_busbar_traces(fig_on)
    fig_on.update_layout(
        title=f"Turn-ON Differential Voltage (t = {t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Differential Voltage (V)', zaxis=dict(range=[0, v_drive * 2.1]))
    )
    st.plotly_chart(fig_on, use_container_width=True)

with col2:
    fig_off = go.Figure(data=[go.Surface(z=V_off, x=x_grid, y=y_grid, colorscale="Viridis")])
    add_busbar_traces(fig_off)
    fig_off.update_layout(
        title=f"Turn-OFF Differential Voltage (t = {t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Differential Voltage (V)', zaxis=dict(range=[0, v_drive * 2.1]))
    )
    st.plotly_chart(fig_off, use_container_width=True)

# --- METRICS ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Busbars Configured", len(configured_busbars))
m2.metric("Center Turn-ON Voltage", f"{V_on[resolution//2, resolution//2]:.2f} V")
m3.metric("Center Turn-OFF Voltage", f"{V_off[resolution//2, resolution//2]:.2f} V")
m4.metric("FDTD Iteration Steps", f"{total_steps:,}")