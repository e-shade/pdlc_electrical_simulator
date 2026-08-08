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
st.caption("Coupled top/bottom ITO layer finite-difference solver with robust boundary safeguarding and ESR modeling.")

# --- SIDEBAR GUI ---
with st.sidebar:
    st.header("1. Film Geometry")
    film_w_cm = st.slider("Width (cm)", 10.0, 300.0, 60.0, step=5.0)
    film_l_cm = st.slider("Length (cm)", 10.0, 300.0, 60.0, step=5.0)
    
    film_w = film_w_cm / 100.0
    film_l = film_l_cm / 100.0

    st.header("2. Electrical & Bus Bar Properties")
    r_sq = st.number_input("ITO Sheet Resistance (Ω/sq)", min_value=10.0, max_value=2000.0, value=70.0, step=10.0)
    c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, value=11.0, step=0.5)
    c_area = c_area_uf * 1e-6
    
    busbar_tape_width_cm = st.number_input("Bus Bar Tape Width (cm)", min_value=0.2, max_value=10.0, value=1.0, step=0.1)
    busbar_sheet_res = st.number_input("Bus Bar Material Sheet Resistance (mΩ/sq)", min_value=0.1, max_value=100.0, value=5.0, step=0.5) * 1e-3
    
    v_rms = st.slider("Applied AC Voltage (RMS)", 10.0, 150.0, 48.0, step=2.0)
    v_peak = v_rms * np.sqrt(2)
    st.caption(f"Peak Voltage ($V_p$): **{v_peak:.1f} V**")

    st.header("3. Multi-Busbar Configuration")
    if 'busbars' not in st.session_state:
        st.session_state.busbars = [
            {'side': 'Top', 'start_pos': 0.0, 'length': film_w_cm, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
            {'side': 'Bottom', 'start_pos': 0.0, 'length': film_w_cm, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
        ]

    col_add, col_rem = st.columns(2)
    if col_add.button("➕ Add Busbar"):
        st.session_state.busbars.append(
            {'side': 'Left', 'start_pos': 0.0, 'length': film_l_cm / 2.0, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
        )
    if col_rem.button("🗑️ Remove Last") and len(st.session_state.busbars) > 1:
        st.session_state.busbars.pop()

    configured_busbars = []
    for i, bb in enumerate(st.session_state.busbars):
        current_signal = st.session_state.get(f"sig_{i}", bb['signal'])
        current_layer = st.session_state.get(f"lay_{i}", bb['layer'])
        current_side = st.session_state.get(f"side_{i}", bb['side'])

        with st.expander(f"Busbar #{i+1} ({current_layer})", expanded=True):
            if current_signal == 'Positive (+)':
                st.markdown(":red[**● Signal: Positive (+) [Red Wireframe]**]")
            else:
                st.markdown(":blue[**● Signal: Negative (-) [Blue Wireframe]**]")

            col_s, col_l = st.columns(2)
            side = col_s.selectbox(f"Side #{i+1}", ["Top", "Bottom", "Left", "Right"], index=["Top", "Bottom", "Left", "Right"].index(current_side), key=f"side_{i}")
            layer = col_l.selectbox(f"Layer #{i+1}", ["Top ITO", "Bottom ITO"], index=0 if current_layer=='Top ITO' else 1, key=f"lay_{i}")

            max_edge_len = film_w_cm if side in ["Top", "Bottom"] else film_l_cm
            
            # --- ROBUST INPUT SAFEGUARDING ---
            # Automatically clamp start position and length so they never exceed edge bounds
            safe_start = min(float(bb['start_pos']), max_edge_len - 0.1)
            max_allowed_len = max(0.1, max_edge_len - safe_start)
            safe_len = min(float(bb['length']), max_allowed_len)

            col_p1, col_p2 = st.columns(2)
            start_pos = col_p1.number_input(f"Start Pos (cm) #{i+1}", 0.0, max(0.0, max_edge_len - 0.1), safe_start, step=1.0, key=f"start_{i}")
            
            # Dynamically restrict the length maximum based on the chosen start position
            remaining_max_len = max(0.1, max_edge_len - start_pos)
            length_val = col_p2.number_input(f"Length (cm) #{i+1}", 0.1, remaining_max_len, min(safe_len, remaining_max_len), step=1.0, key=f"len_{i}")
            
            _, col_sig = st.columns(2)
            signal = col_sig.selectbox(f"Signal #{i+1}", ["Positive (+)", "Negative (-)"], index=0 if current_signal=='Positive (+)' else 1, key=f"sig_{i}")

            # Translate Side + Start/Length into spatial bounding box coordinates (in meters)
            tape_w_m = busbar_tape_width_cm / 100.0
            start_m = start_pos / 100.0
            end_m = (start_pos + length_val) / 100.0

            if side == 'Top':
                x_min, x_max = start_m, end_m
                y_min, y_max = 0.0, tape_w_m
            elif side == 'Bottom':
                x_min, x_max = start_m, end_m
                y_min, y_max = film_l - tape_w_m, film_l
            elif side == 'Left':
                x_min, x_max = 0.0, tape_w_m
                y_min, y_max = start_m, end_m
            else:  # Right
                x_min, x_max = film_w - tape_w_m, film_w
                y_min, y_max = start_m, end_m

            configured_busbars.append({
                'side': side, 'start_pos': start_pos, 'length': length_val,
                'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                'layer': layer, 'signal': signal
            })

    st.header("4. Simulation Settings")
    t_snapshot_us = st.slider("Snapshot Time (µs)", 10, 5000, 230, step=10)
    t_snapshot = t_snapshot_us * 1e-6
    resolution = st.select_slider("Grid Resolution", options=[20, 40, 60], value=40)

# --- COUPLED FDTD SOLVER ---
def simulate_all_profiles(C_A, R_sq, tape_w_cm, copper_r_sq, width, length, busbars, V_peak, v_rms_val, t_snap, nx, ny):
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

    final_t_on, final_b_on = run_fdtd(np.zeros((ny, nx)), np.zeros((ny, nx)), t_vals, b_vals, steps)
    V_eff_on = np.abs(final_t_on - final_b_on) / np.sqrt(2)

    steady_t, steady_b = run_fdtd(np.zeros((ny, nx)), np.zeros((ny, nx)), t_vals, b_vals, int(5000e-6 / dt))
    final_t_off, final_b_off = run_fdtd(steady_t, steady_b, np.zeros((ny, nx)), np.zeros((ny, nx)), steps)
    V_eff_off = np.abs(final_t_off - final_b_off) / np.sqrt(2)

    V_eff_steady = np.abs(steady_t - steady_b) / np.sqrt(2)

    # --- TAPE-WIDTH DEPENDENT ESR POWER CALCULATIONS ---
    total_area = width * length
    f_driver = 60.0  # AC driving frequency
    
    total_capacitance = C_A * total_area
    reactive_current_rms = 2 * np.pi * f_driver * total_capacitance * v_rms_val
    apparent_power_va = v_rms_val * reactive_current_rms
    
    base_pf = 0.083 + (0.441 * total_area)
    resistance_scaling = 30.0 / max(R_sq, 5.0)
    effective_pf = np.clip(base_pf * (0.5 + 0.5 * resistance_scaling), 0.05, 0.85)
    core_active_power = apparent_power_va * effective_pf
    
    tape_w_m = tape_w_cm / 100.0
    total_busbar_esr = 0.0
    for bb in busbars:
        bb_length = max((bb['x_max'] - bb['x_min']), (bb['y_max'] - bb['y_min']))
        if tape_w_m > 0:
            total_busbar_esr += copper_r_sq * (bb_length / tape_w_m)
            
    busbar_esr_power = (reactive_current_rms ** 2) * total_busbar_esr
    active_power_w = core_active_power + busbar_esr_power

    return V_eff_on, V_eff_off, V_eff_steady, steps, active_power_w, active_power_w

# --- EXECUTE ---
with st.spinner("Solving transient and steady-state fields..."):
    V_on, V_off, V_steady, total_steps, active_p_w, dc_power_w = simulate_all_profiles(
        c_area, r_sq, busbar_tape_width_cm, busbar_sheet_res, film_w, film_l, configured_busbars, v_peak, v_rms, t_snapshot, resolution, resolution
    )

# --- PLOTTING ---
x_grid_cm = np.linspace(0, film_w_cm, resolution)
y_grid_cm = np.linspace(0, film_l_cm, resolution)

def add_busbar_traces(fig):
    for i, bb in enumerate(configured_busbars):
        bx = [bb['x_min'] * 100.0, bb['x_max'] * 100.0, bb['x_max'] * 100.0, bb['x_min'] * 100.0, bb['x_min'] * 100.0]
        by = [bb['y_min'] * 100.0, bb['y_min'] * 100.0, bb['y_max'] * 100.0, bb['y_max'] * 100.0, bb['y_min'] * 100.0]
        bz = [v_rms * 1.05] * 5
        color = "red" if bb['signal'] == 'Positive (+)' else "blue"
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color=color, width=6),
            showlegend=False
        ))

# Row 1: Transient Maps Side-by-Side
col1, col2 = st.columns(2)

with col1:
    fig_on = go.Figure(data=[go.Surface(z=V_on, x=x_grid_cm, y=y_grid_cm, colorscale="Inferno")])
    add_busbar_traces(fig_on)
    fig_on.update_layout(
        title=f"Turn-ON Differential Voltage RMS ({t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (cm)', yaxis_title='Length (cm)', zaxis_title='RMS Voltage (V)', zaxis=dict(range=[0, v_rms * 1.1]))
    )
    st.plotly_chart(fig_on, use_container_width=True)

with col2:
    fig_off = go.Figure(data=[go.Surface(z=V_off, x=x_grid_cm, y=y_grid_cm, colorscale="Viridis")])
    add_busbar_traces(fig_off)
    fig_off.update_layout(
        title=f"Turn-OFF Differential Voltage RMS ({t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='RMS Voltage (V)', zaxis=dict(range=[0, v_rms * 1.1]))
    )
    st.plotly_chart(fig_off, use_container_width=True)

# Row 2: Steady-State Map Below
st.markdown("---")
st.subheader("Steady-State ON State Distribution")
fig_steady = go.Figure(data=[go.Surface(z=V_steady, x=x_grid_cm, y=y_grid_cm, colorscale="Plasma")])
add_busbar_traces(fig_steady)
fig_steady.update_layout(
    title="Steady-State ON State Differential Voltage RMS",
    autosize=True, margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='RMS Voltage (V)', zaxis=dict(range=[0, v_rms * 1.1]))
)
st.plotly_chart(fig_steady, use_container_width=True)

# --- METRICS ---
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Busbars", len(configured_busbars))
m2.metric("Center Turn-ON (RMS)", f"{V_on[resolution//2, resolution//2]:.2f} V")
m3.metric("Center Turn-OFF (RMS)", f"{V_off[resolution//2, resolution//2]:.2f} V")
m4.metric("Steady-State Center (RMS)", f"{V_steady[resolution//2, resolution//2]:.2f} V")
m5.metric("Est. DC Supply Power Draw", f"{active_p_w:.3f} W")