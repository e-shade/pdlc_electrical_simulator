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
st.caption("Coupled top/bottom ITO layer finite-difference solver with AC RMS voltage metrics and resistance-sensitive power estimation.")

# --- SIDEBAR GUI ---
with st.sidebar:
    st.header("1. Film Geometry")
    film_w_cm = st.slider("Width (cm)", 10.0, 300.0, 100.0, step=5.0)
    film_l_cm = st.slider("Length (cm)", 10.0, 300.0, 100.0, step=5.0)
    
    # Convert cm to meters for internal solver calculations
    film_w = film_w_cm / 100.0
    film_l = film_l_cm / 100.0

    st.header("2. Electrical Properties")
    r_sq = st.number_input("ITO Sheet Resistance (Ω/sq)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, value=11.0, step=0.5)
    c_area = c_area_uf * 1e-6
    
    v_rms = st.slider("Applied AC Voltage (RMS)", 10.0, 150.0, 48.0, step=2.0)
    v_peak = v_rms * np.sqrt(2)
    st.caption(f"Peak Voltage ($V_p$): **{v_peak:.1f} V**")

    st.header("3. Multi-Busbar Configuration")
    if 'busbars' not in st.session_state:
        st.session_state.busbars = [
            {'x_start': 0.0, 'x_end': 5.0, 'y_start': 0.0, 'y_end': 40.0, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
            {'x_start': 0.0, 'x_end': 5.0, 'y_start': 60.0, 'y_end': 100.0, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
        ]

    col_add, col_rem = st.columns(2)
    if col_add.button("➕ Add Busbar"):
        st.session_state.busbars.append(
            {'x_start': 0.0, 'x_end': film_w_cm, 'y_start': 0.0, 'y_end': film_l_cm / 2.0, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
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
            x_start_cm = col_x1.number_input(f"X start (cm) #{i+1}", 0.0, film_w_cm, float(bb['x_start']), step=1.0, key=f"x1_{i}")
            x_end_cm = col_x2.number_input(f"X end (cm) #{i+1}", 0.0, film_w_cm, float(bb['x_end']), step=1.0, key=f"x2_{i}")
            
            col_y1, col_y2 = st.columns(2)
            y_start_cm = col_y1.number_input(f"Y start (cm) #{i+1}", 0.0, film_l_cm, float(bb['y_start']), step=1.0, key=f"y1_{i}")
            y_end_cm = col_y2.number_input(f"Y end (cm) #{i+1}", 0.0, film_l_cm, float(bb['y_end']), step=1.0, key=f"y2_{i}")
            
            col_l, col_s = st.columns(2)
            layer = col_l.selectbox(f"Layer #{i+1}", ["Top ITO", "Bottom ITO"], index=0 if current_layer=='Top ITO' else 1, key=f"lay_{i}")
            signal = col_s.selectbox(f"Signal #{i+1}", ["Positive (+)", "Negative (-)"], index=0 if current_signal=='Positive (+)' else 1, key=f"sig_{i}")
            
            configured_busbars.append({
                'x_min': min(x_start_cm, x_end_cm) / 100.0, 'x_max': max(x_start_cm, x_end_cm) / 100.0,
                'y_min': min(y_start_cm, y_end_cm) / 100.0, 'y_max': max(y_start_cm, y_end_cm) / 100.0,
                'layer': layer, 'signal': signal
            })

    st.header("4. Simulation Settings")
    t_snapshot_us = st.slider("Snapshot Time (µs)", 10, 5000, 500, step=10)
    t_snapshot = t_snapshot_us * 1e-6
    resolution = st.select_slider("Grid Resolution", options=[20, 40, 60], value=40)

# --- COUPLED FDTD SOLVER ---
@st.cache_data
def simulate_all_profiles(C_A, R_sq, width, length, busbars, V_peak, v_rms_val, t_snap, nx, ny):
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

    # --- RESISTANCE-SENSITIVE POWER CALCULATIONS ---
    total_area = width * length
    total_capacitance = C_A * total_area
    f_driver = 60.0  # AC driving frequency
    
    reactive_current_rms = 2 * np.pi * f_driver * total_capacitance * v_rms_val
    apparent_power_va = v_rms_val * reactive_current_rms
    
    base_pf = min(0.524, 0.08 + (0.444 * total_area))
    resistance_scaling_factor = np.clip(R_sq / 150.0, 0.5, 2.0)
    empirical_pf = min(0.85, base_pf * resistance_scaling_factor)
    
    grad_y_t, grad_x_t = np.gradient(steady_t, dy, dx)
    grad_y_b, grad_x_b = np.gradient(steady_b, dy, dx)
    joule_dissipation = ((grad_x_t**2 + grad_y_t**2) + (grad_x_b**2 + grad_y_b**2)) / R_sq
    active_joule_power = np.sum(joule_dissipation) * (dx * dy)
    
    active_power_w = (apparent_power_va * empirical_pf) + active_joule_power
    dc_supply_power_w = active_power_w + (apparent_power_va * 0.10)

    return V_eff_on, V_eff_off, V_eff_steady, steps, active_power_w, dc_supply_power_w

# --- EXECUTE ---
with st.spinner("Solving transient and steady-state fields..."):
    V_on, V_off, V_steady, total_steps, active_p_w, dc_power_w = simulate_all_profiles(
        c_area, r_sq, film_w, film_l, configured_busbars, v_peak, v_rms, t_snapshot, resolution, resolution
    )

# --- PLOTTING (Axis labels scaled to cm) ---
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
        scene=dict(xaxis_title='Width (cm)', yaxis_title='Length (cm)', zaxis_title='RMS Voltage (V)', zaxis=dict(range=[0, v_rms * 1.1]))
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
    scene=dict(xaxis_title='Width (cm)', yaxis_title='Length (cm)', zaxis_title='RMS Voltage (V)', zaxis=dict(range=[0, v_rms * 1.1]))
)
st.plotly_chart(fig_steady, use_container_width=True)

# --- METRICS ---
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Busbars", len(configured_busbars))
m2.metric("Center Turn-ON (RMS)", f"{V_on[resolution//2, resolution//2]:.2f} V")
m3.metric("Center Turn-OFF (RMS)", f"{V_off[resolution//2, resolution//2]:.2f} V")
m4.metric("Steady-State Center (RMS)", f"{V_steady[resolution//2, resolution//2]:.2f} V")
m5.metric("Est. DC Supply Power Draw", f"{dc_power_w:.3f} W")