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
st.caption("Coupled top/bottom ITO layer finite-difference solver with custom multi-busbar routing and 3D visual markers.")

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
    st.write("Configure individual busbar strips across the film boundaries:")
    
    # Initialize session state for dynamic busbars if not present
    if 'busbars' not in st.session_state:
        st.session_state.busbars = [
            {'x_start': 0.0, 'x_end': 0.05, 'y_start': 0.0, 'y_end': film_l, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
            {'x_start': film_w - 0.05, 'x_end': film_w, 'y_start': 0.0, 'y_end': film_l, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
        ]

    col_add, col_rem = st.columns(2)
    if col_add.button("➕ Add Busbar"):
        st.session_state.busbars.append(
            {'x_start': 0.0, 'x_end': film_w, 'y_start': 0.0, 'y_end': 0.05, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
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

# --- COUPLED 2-LAYER FDTD SOLVER (BIPOLAR SIGNALS) ---
@st.cache_data
def simulate_coupled_transients(C_A, R_sq, width, length, busbars, V_mag, t_snap, nx, ny):
    dx = width / (nx - 1)
    dy = length / (ny - 1)
    
    dt = 0.2 * (R_sq * C_A) / (1/(dx**2) + 1/(dy**2))
    steps = int(t_snap / dt)
    
    alpha = dt / (R_sq * C_A * dx**2)

    def apply_busbar_masks(V_top, V_bot):
        top_pos_mask, top_val_mask = np.zeros((ny, nx), dtype=bool), np.zeros((ny, nx))
        bot_pos_mask, bot_val_mask = np.zeros((ny, nx), dtype=bool), np.zeros((ny, nx))
        
        has_top_bb = False
        has_bot_bb = False

        for bb in busbars:
            ix_min = max(0, int(bb['x_min'] / width * (nx - 1)))
            ix_max = min(nx - 1, int(bb['x_max'] / width * (nx - 1)))
            iy_min = max(0, int(bb['y_min'] / length * (ny - 1)))
            iy_max = min(ny - 1, int(bb['y_max'] / length * (ny - 1)))
            
            # Bipolar mapping: Positive = +V_mag, Negative = -V_mag (or 0 reference)
            val = V_mag if bb['signal'] == 'Positive (+)' else -V_mag
            
            if bb['layer'] == 'Top ITO':
                V_top[iy_min:iy_max+1, ix_min:ix_max+1] = val
                top_pos_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                top_val_mask[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_top_bb = True
            else:
                V_bot[iy_min:iy_max+1, ix_min:ix_max+1] = val
                bot_pos_mask[iy_min:iy_max+1, ix_min:ix_max+1] = True
                bot_val_mask[iy_min:iy_max+1, ix_min:ix_max+1] = val
                has_bot_bb = True
                
        # Fallback defaults if a layer has no busbars assigned
        if not has_top_bb: V_top[:, :] = 0.0
        if not has_bot_bb: V_bot[:, :] = 0.0
                
        return V_top, V_bot, (top_pos_mask, top_val_mask, bot_pos_mask, bot_val_mask)

    V_top = np.zeros((ny, nx))
    V_bot = np.zeros((ny, nx))
    V_top, V_bot, masks = apply_busbar_masks(V_top, V_bot)
    top_pos_mask, top_val_mask, bot_pos_mask, bot_val_mask = masks

    for _ in range(steps):
        V_top_new = V_top.copy()
        V_top_new[1:-1, 1:-1] = V_top[1:-1, 1:-1] + alpha * (
            V_top[1:-1, 2:] + V_top[1:-1, :-2] + V_top[2:, 1:-1] + V_top[:-2, 1:-1] - 4*V_top[1:-1, 1:-1]
        )
        V_bot_new = V_bot.copy()
        V_bot_new[1:-1, 1:-1] = V_bot[1:-1, 1:-1] + alpha * (
            V_bot[1:-1, 2:] + V_bot[1:-1, :-2] + V_bot[2:, 1:-1] + V_bot[:-2, 1:-1] - 4*V_bot[1:-1, 1:-1]
        )
        
        # Enforce Dirichlet boundary conditions for busbars
        V_top_new[top_pos_mask] = top_val_mask[top_pos_mask]
        V_bot_new[bot_pos_mask] = bot_val_mask[bot_pos_mask]
        
        # Neumann boundaries for unconstrained edges
        V_top_new[0, :] = V_top_new[1, :]
        V_top_new[-1, :] = V_top_new[-2, :]
        V_top_new[:, 0] = V_top_new[:, 1]
        V_top_new[:, -1] = V_top_new[:, -2]

        V_bot_new[0, :] = V_bot_new[1, :]
        V_bot_new[-1, :] = V_bot_new[-2, :]
        V_bot_new[:, 0] = V_bot_new[:, 1]
        V_bot_new[:, -1] = V_bot_new[:, -2]

        # Re-apply busbars post Neumann
        V_top_new[top_pos_mask] = top_val_mask[top_pos_mask]
        V_bot_new[bot_pos_mask] = bot_val_mask[bot_pos_mask]

        V_top, V_bot = V_top_new, V_bot_new

    # Effective potential difference across the PDLC dielectric layer (|V_top - V_bot|)
    V_effective = np.abs(V_top - V_bot)
    return V_effective, V_top, V_bot, steps

# --- EXECUTE ---
with st.spinner("Solving multi-layer distributed network..."):
    V_eff, V_t, V_b, total_steps = simulate_coupled_transients(
        c_area, r_sq, film_w, film_l, configured_busbars, v_drive, t_snapshot, resolution, resolution
    )

# --- PLOTTING WITH BUSBAR MARKERS ---
x_grid = np.linspace(0, film_w, resolution)
y_grid = np.linspace(0, film_l, resolution)

def add_busbar_traces(fig, layer_filter_name=None):
    """Adds 3D rectangular strips mapping the configured busbars onto the surface."""
    for i, bb in enumerate(configured_busbars):
        if layer_filter_name and bb['layer'] != layer_filter_name:
            continue
            
        bx = [bb['x_min'], bb['x_max'], bb['x_max'], bb['x_min'], bb['x_min']]
        by = [bb['y_min'], bb['y_min'], bb['y_max'], bb['y_max'], bb['y_min']]
        # Place a raised marker strip at peak voltage level for visibility
        bz = [v_drive * 1.02] * 5 
        
        color = "red" if bb['signal'] == 'Positive (+)' else "blue"
        
        # Add outline strip
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines',
            line=dict(color=color, width=6),
            name=f"BB#{i+1} ({bb['layer']} {bb['signal']})"
        ))

col1, col2 = st.columns(2)

with col1:
    fig_eff = go.Figure(data=[go.Surface(z=V_eff, x=x_grid, y=y_grid, colorscale="Inferno")])
    add_busbar_traces(fig_eff)
    fig_eff.update_layout(
        title=f"Effective Dielectric Voltage Drop & Busbars (t = {t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Voltage (V)', zaxis=dict(range=[0, v_drive * 1.1]))
    )
    st.plotly_chart(fig_eff, use_container_width=True)

with col2:
    fig_top = go.Figure(data=[go.Surface(z=V_t, x=x_grid, y=y_grid, colorscale="Viridis")])
    add_busbar_traces(fig_top, layer_filter_name="Top ITO")
    fig_top.update_layout(
        title=f"Top ITO Potential & Top Busbars (t = {t_snapshot_us} µs)",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Voltage (V)', zaxis=dict(range=[0, v_drive * 1.1]))
    )
    st.plotly_chart(fig_top, use_container_width=True)

# --- METRICS ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Busbars Configured", len(configured_busbars))
m2.metric("Center Dielectric Voltage", f"{V_eff[resolution//2, resolution//2]:.2f} V")
m3.metric("Max Surface Voltage Gradient", f"{np.max(V_eff) - np.min(V_eff):.2f} V")
m4.metric("FDTD Iteration Steps", f"{total_steps:,}")