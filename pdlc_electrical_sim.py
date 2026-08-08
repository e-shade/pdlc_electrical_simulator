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

st.title("⚡ Advanced Multi-Busbar PDLC Transient & Steady-State Simulator")
st.caption("Coupled top/bottom ITO layer finite-difference solver with steady-state mapping and power estimation.")

# --- SIDEBAR GUI ---
with st.sidebar:
    st.header("1. Film Geometry")
    film_w = st.slider("Width (m)", 0.1, 3.0, 1.0, step=0.1)
    film_l = st.slider("Length (m)", 0.1, 3.0, 1.0, step=0.1)

    st.header("2. Electrical Properties")
    r_sq = st.number_input("ITO Sheet Resistance (Ω/sq)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, value=11.0, step=0.5)
    c_area = c_area_uf * 1e-6
    
    v_rms = st.slider("Applied AC Voltage (RMS)", 10.0, 150.0, 48.0, step=2.0)
    v_peak = v_rms * np.sqrt(2)
    st.caption(f"Equivalent Peak Voltage ($V_p$): **{v_peak:.1f} V**")

    st.header("3. Multi-Busbar Configuration")
    if 'busbars' not in st.session_state:
        st.session_state.busbars = [
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.0, 'y_end': 0.4, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.6, 'y_end': 1.0, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
        ]

    col_add, col_rem = st.columns(2)
    if col_add.button("➕ Add Busbar"):
        st.session_state.busbars.append(
            {'x_start': 0.0, 'x_end': film_w, 'y_start': 0.0, 'y_end': 0.5, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
        )
    if col_rem.button("🗑️ Remove Last") and len(st.session_state.busbars) > 1:
        st.session_state.busbars.pop()

    configured_busbars = []
    for i, bb in enumerate(st.session_state.busbars):
        current_signal = st.session_state.get(f"sig_{i}", bb['signal'])
        current_layer = st.session_state.get(f"lay_{i}", bb['layer'])

        with st.expander(f"Busbar #{i+1} ({current_layer})", expanded=True):
            if current_signal == 'Positive (+)':
                st.markdown(":red[**● Signal: Positive (+) [Red Wireframe]**]")
            else:
                st.markdown(":blue[**● Signal: Negative (-) [Blue Wireframe]**]")

            col_x1, col_x2 = st.columns(2)
            x_start = col_x1.number_input(f"X start (m) #{i+1}", 0.0, film_w, float(bb['x_start']), step=0.05, key=f"x1_{i}")
            x_end = col_x2.number_input(f"X end (m) #{i+1}", 0.0, film_w, float(bb['x_end']), step=0.05, key=f"x2_{i}")
            
            col_y1, col_y2 = st.columns(2)
            y_start = col_y1.number_input(f"Y start (m) #{i+1}", 0.0, film_l, float(bb['y_start']), step=0.05, key=f"y1_{i}")
            y_end = col_y2.number_input(f"Y end (m) #{i+1}", 0.0, film_l, float(bb['y_end']), step=0.05, key=f"y2_{i}")
            
            col_l, col_s = st.columns(2)
            layer = col_l.selectbox(f"Layer #{i+1}", ["Top ITO", "Bottom ITO"], index=0 if current_layer=='Top ITO' else 1, key=f"lay_{i}")
            signal = col_s.selectbox(f"Signal #{i+1}", ["Positive (+)", "Negative (-)"], index=0 if current_signal=='Positive (+)' else 1, key=f"sig_{i}")
            
            configured_busbars.append({
                'x_min': min(x_start, x_end), 'x_max': max(x_start, x_end),
                'y_min': min(y_start, y_end), 'y_max': max(y_start, y_end),
                'layer': layer, 'signal': signal
            })

    st.header("4. Simulation Settings")
    t_snapshot_us = st.slider("Snapshot Time (µs)", 10, 5000, 500, step=10)
    t_snapshot = t_snapshot_us * 1e-6
    resolution = st.select_slider("Grid Resolution", options=[20, 40, 60], value=40)

# --- COUPLED FDTD SOLVER ---
@st.cache_data
def simulate_all_profiles(C_A, R_sq, width, length, busbars, V_peak, t_snap, nx, ny):
    dx = width / (nx - 1)
    dy = length / (ny - 1)
    
    dt = 0.2 * (R_sq * C_A) / (1/(dx**2) + 1/(dy**2))
    steps = int(t_snap / dt)
    alpha = dt / (R_sq * C_A * dx**2)

    def get_masks_and_targets(ny, nx):
        top_mask = np.zeros((ny, nx), dtype=bool)
        top_vals = np.zeros((ny, nx))
        bot_mask = np.zeros((ny, nx), dtype=bool)
        bot_vals = np.zeros((ny, nx))
        
        has_top, has_bot = False, False
        for bb in busbars:
            ix_min = max(0, int(bb['x_min'] / width * (nx - 1)))
            ix_max = min(nx - 1, int(bb['x_max'] / width * (nx - 1)))
            iy_min = max(0, int(bb['y_min'] / length * (ny - 1)))
            iy_max = min(ny - 1, int(bb['y_max'] / length * (ny - 1)))
            
            val = (V_peak / 2.0) if bb['signal'] == 'Positive (+)' else (-V_peak / 2.0)
            
            if bb['layer'] == 'Top ITO':
                top_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                top_vals[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_top = True
            else:
                bot_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                bot_vals[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_bot = True
                
        if has_top and not has_bot:
            bot_mask[:, :] = True
            bot_vals[:, :] = 0.0
        elif has_bot and not has_top:
            top_mask[:, :] = True
            top_vals[:, :] = 0.0
                
        return top_mask, top_vals, bot_mask, bot_vals

    t_mask, t_vals, b_mask, b_vals = get_masks_and_targets(ny, nx)

    def run_fdtd(init_t, init_b, target_t, target_b, custom_steps):
        V_t, V_b = init_t.copy(), init_b.copy()
        V_t[t_mask] = target_t[t_mask]
        V_b[b_mask] = target_b[b_mask]
        
        for _ in range(custom_steps):
            V_t_new = V_t.copy()
            V_t_new[1:-1, 1:-1] = V_t[1:-1, 1:-1] + alpha * (
                V_t[1:-1, 2:] + V_t[1:-1, :-2] + V_t[2:, 1:-1] + V_t[:-2, 1:-1] - 4*V_t[1:-1, 1:-1]
            )
            V_b_new = V_b.copy()
            V_b_new[1:-1, 1:-1] = V_b[1:-1, 1:-1] + alpha * (
                V_b[1:-1, 2:] + V_b[1:-1, :-2] + V_b[2:, 1:-1] + V_b[:-2, 1:-1] - 4*V_b[1:-1, 1:-1]
            )
            
            V_t_new[t_mask] = target_t[t_mask]
            V_b_new[b_mask] = target_b[b_mask]
            
            for V_arr in [V_t_new, V_b_new]:
                V_arr[0, :] = V_arr[1, :]
                V_arr[-1, :] = V_arr[-2, :]
                V_arr[:, 0] = V_arr[:, 1]
                V_arr[:, -1] = V_arr[:, -2]
                
            V_t_new[t_mask] = target_t[t_mask]
            V_b_new[b_mask] = target_b[b_mask]
            
            V_t, V_b = V_t_new, V_b_new
        return V_t, V_b

    # 1. Turn-ON Transient
    final_t_on, final_b_on = run_fdtd(np.zeros((ny, nx)), np.zeros((ny, nx)), t_vals, b_vals, steps)
    V_eff_on = np.abs(final_t_on - final_b_on)

    # 2. Turn-OFF Transient
    steady_t, steady_b = run_fdtd(np.zeros((ny, nx)), np.zeros((ny, nx)), t_vals, b_vals, int(5000e-6 / dt))
    final_t_off, final_b_off = run_fdtd(steady_t, steady_b, np.zeros((ny, nx)), np.zeros((ny, nx)), steps)
    V_eff_off = np.abs(final_t_off - final_b_off)

    # 3. Steady-State ON State Map
    V_eff_steady = np.abs(steady_t - steady_b)

    # Power estimation: Active power loss in resistive ITO layers + Capacitive charging current estimate
    # P_active = V_rms^2 / R_eff approximation
    total_film_area = width * length
    approx_resistance = R_sq * (length / width if width > 0 else 1.0)
    active_power_w = (v_rms ** 2) / max(approx_resistance, 1.0)

    return V_eff_on, V_eff_off, V_eff_steady, steps, active_power_w

# --- EXECUTE ---
with st.spinner("Solving transient and steady-state fields..."):
    V_on, V_off, V_steady, total_steps, power_w = simulate_all_profiles(
        c_area, r_sq, film_w, film_l, configured_busbars, v_peak, t_snapshot, resolution, resolution
    )

# --- PLOTTING ---
x_grid = np.linspace(0, film_w, resolution)
y_grid = np.linspace(0, film_l, resolution)

def add_busbar_traces(fig):
    for i, bb in enumerate(configured_busbars):
        bx = [bb['x_min'], bb['x_max'], bb['x_max'], bb['x_min'], bb['x_min']]
        by = [bb['y_min'], bb['y_min'], bb['y_max'], bb['y_max'], bb['y_min']]
        bz = [v_peak * 1.05] * 5
        color = "red" if bb['signal'] == 'Positive (+)' else "blue"
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color=color, width=6),
            showlegend=False
        ))

col1, col2, col3 = st.columns(3)

with col1:
    fig_on = go.Figure(data=[go.Surface(z=V_on, x=x_grid, y=y_grid, colorscale="Inferno")])
    add_busbar_traces(fig_on)
    fig_on.update_layout(
        title=f"Turn-ON ({t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Voltage (V)', zaxis=dict(range=[0, v_peak * 1.1]))
    )
    st.plotly_chart(fig_on, use_container_width=True)

with col2:
    fig_off = go.Figure(data=[go.Surface(z=V_off, x=x_grid, y=y_grid, colorscale="Viridis")])
    add_busbar_traces(fig_off)
    fig_off.update_layout(
        title=f"Turn-OFF ({t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Voltage (V)', zaxis=dict(range=[0, v_peak * 1.1]))
    )
    st.plotly_chart(fig_off, use_container_width=True)

with col3:
    fig_steady = go.Figure(data=[go.Surface(z=V_steady, x=x_grid, y=y_grid, colorscale="Plasma")])
    add_busbar_traces(fig_steady)
    fig_steady.update_layout(
        title="Steady-State ON State",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Voltage (V)', zaxis=dict(range=[0, v_peak * 1.1]))
    )
    st.plotly_chart(fig_steady, use_container_width=True)

# --- METRICS ---
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Busbars", len(configured_busbars))
m2.metric("Center Turn-ON Voltage", f"{V_on[resolution//2, resolution//2]:.2f} V")
m3.metric("Center Turn-OFF Voltage", f"{V_off[resolution//2, resolution//2]:.2f} V")
m4.metric("Steady-State Center Voltage", f"{V_steady[resolution//2, resolution//2]:.2f} V")
m5.metric("Est. Active Power (RMS)", f"{power_w:.3f} W")