import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Busbar PDLC Simulator", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { min-width: 520px !important; }
    
    /* Robust CSS rules for centering KaTeX display/block equations */
    .katex-display {
        display: block !important;
        text-align: center !important;
        margin: 1.2em 0 !important;
    }
    .katex-display > .katex {
        display: inline-block !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE DEFAULTS ---
defaults = {
    "width_cm": 60.0,
    "length_cm": 60.0,
    "r_sq": 70.0,
    "c_area_uf": 11.0,
    "tape_width_cm": 1.0,
    "sheet_res_m_ohms": 5.0,
    "v_rms": 48.0,
    "v_threshold": 15.0,
    "frequency": 60.0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if 'busbars' not in st.session_state:
    st.session_state.busbars = [
        {'side': 'Top', 'start_pos': 0.0, 'length': 60.0, 'layer': 'Top ITO', 'signal': 'Positive (+)'},
        {'side': 'Bottom', 'start_pos': 0.0, 'length': 60.0, 'layer': 'Bottom ITO', 'signal': 'Negative (-)'}
    ]

st.title("⚡ Multi-Busbar PDLC Transient & Steady-State Simulator")

# --- HELP MODAL DIALOG ---
@st.dialog("📖 Simulator Help & Documentation", width="large")
def show_help_dialog():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("README.md file not found in the application directory.")

# --- COUPLED FDTD SOLVER ---
def simulate_all_profiles(C_A, R_sq, tape_w_cm, copper_r_sq, width, length, busbars, V_peak, v_rms_val, freq_val, t_snap, nx, ny):
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

    total_area = width * length
    f_driver = freq_val  
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

# --- SIDEBAR GUI SPLIT INTO TWO COLUMNS SIDE-BY-SIDE ---
with st.sidebar:
    sb_col1, sb_col2 = st.columns(2)
    
    with sb_col1:
        st.header("⚙️ Configuration Snapshot")
        
        current_settings = {
            "width_cm": st.session_state.width_cm,
            "length_cm": st.session_state.length_cm,
            "r_sq": st.session_state.r_sq,
            "c_area_uf": st.session_state.c_area_uf,
            "tape_width_cm": st.session_state.tape_width_cm,
            "sheet_res_m_ohms": st.session_state.sheet_res_m_ohms,
            "v_rms": st.session_state.v_rms,
            "v_threshold": st.session_state.v_threshold,
            "frequency": st.session_state.frequency,
            "busbars": st.session_state.busbars
        }
        
        custom_filename = st.text_input("Configuration File Name", value="pdlc_config")
        
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)

        if st.button("💾 SAVE TO PRESETS", use_container_width=True):
            clean_filename = custom_filename.strip()
            if not clean_filename.endswith(".json"):
                clean_filename += ".json"
            
            file_path = os.path.join(config_dir, clean_filename)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(current_settings, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
                st.session_state.last_loaded_preset = clean_filename
                st.success(f"Saved & updated `{clean_filename}`!")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving file: {e}")

        # --- PRESET LIBRARY FROM GITHUB CONFIG FOLDER (Auto-loads on selection) ---
        st.subheader("📚 Presets")
        preset_files = [f for f in os.listdir(config_dir) if f.endswith(".json")]

        if preset_files:
            if "last_loaded_preset" not in st.session_state:
                st.session_state.last_loaded_preset = None

            selected_preset = st.selectbox("Config Presets", preset_files, key="preset_selectbox", label_visibility="collapsed")
            
            if selected_preset and selected_preset != st.session_state.last_loaded_preset:
                try:
                    preset_path = os.path.join(config_dir, selected_preset)
                    with open(preset_path, "r") as f:
                        loaded_config = json.load(f)
                    
                    st.session_state.width_cm = float(loaded_config.get("width_cm", 60.0))
                    st.session_state.length_cm = float(loaded_config.get("length_cm", 60.0))
                    st.session_state.r_sq = float(loaded_config.get("r_sq", 70.0))
                    st.session_state.c_area_uf = float(loaded_config.get("c_area_uf", 11.0))
                    st.session_state.tape_width_cm = float(loaded_config.get("tape_width_cm", 1.0))
                    st.session_state.sheet_res_m_ohms = float(loaded_config.get("sheet_res_m_ohms", 5.0))
                    st.session_state.v_rms = float(loaded_config.get("v_rms", 48.0))
                    st.session_state.v_threshold = float(loaded_config.get("v_threshold", 15.0))
                    st.session_state.frequency = float(loaded_config.get("frequency", 60.0))
                    
                    if "busbars" in loaded_config:
                        st.session_state.busbars = loaded_config["busbars"]
                        for k in [k for k in st.session_state.keys() if k.startswith(('side_', 'lay_', 'start_', 'len_', 'sig_'))]:
                            del st.session_state[k]
                            
                        for idx, bb in enumerate(st.session_state.busbars):
                            st.session_state[f"side_{idx}"] = bb.get('side', 'Top')
                            st.session_state[f"lay_{idx}"] = bb.get('layer', 'Top ITO')
                            st.session_state[f"start_{idx}"] = float(bb.get('start_pos', 0.0))
                            st.session_state[f"len_{idx}"] = float(bb.get('length', 10.0))
                            st.session_state[f"sig_{idx}"] = bb.get('signal', 'Positive (+)')

                    st.session_state.last_loaded_preset = selected_preset
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.caption("No files in `config/`.")

        # --- HELP BUTTON ---
        if st.button("❓ HELP", use_container_width=True):
            show_help_dialog()

        st.divider()

        st.header("1. Sim Settings")
        fine_discrete_options = [
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90,
            100, 120, 140, 160, 180, 200, 220, 230, 250, 300, 350, 400, 450,
            500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000
        ]
        t_snapshot_us = st.select_slider(
            "Snapshot Time",
            options=fine_discrete_options,
            value=230,
            format_func=lambda x: f"{x} µs" if x < 1000 else f"{x/1000:.1f} ms"
        )
        t_snapshot = t_snapshot_us * 1e-6
        
        resolution = st.slider("Grid Resolution", min_value=5, max_value=40, value=10, step=1)

        # Transient End Time Slider (starts at t=0 up to specified end ms)
        transient_end_ms = st.slider(
            "Transient End Time (ms)",
            min_value=1.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

        st.header("2. Film Geometry")
        film_w_cm = st.slider("Width (cm)", 10.0, 300.0, key="width_cm", step=5.0)
        film_l_cm = st.slider("Length (cm)", 10.0, 300.0, key="length_cm", step=5.0)
        
        film_w = film_w_cm / 100.0
        film_l = film_l_cm / 100.0

    with sb_col2:
        st.header("3. Electrical Props")
        r_sq = st.number_input("ITO Res (Ω/sq)", min_value=10.0, max_value=2000.0, key="r_sq", step=10.0)
        c_area_uf = st.number_input("Capacitance (µF/m²)", min_value=0.1, max_value=100.0, key="c_area_uf", step=0.5)
        c_area = c_area_uf * 1e-6
        
        busbar_tape_width_cm = st.number_input("Tape Width (cm)", min_value=0.2, max_value=10.0, key="tape_width_cm", step=0.1)
        busbar_sheet_res = st.number_input("Bar Res (mΩ/sq)", min_value=0.1, max_value=100.0, key="sheet_res_m_ohms", step=0.5) * 1e-3
        
        v_rms = st.slider("Voltage (RMS)", 10.0, 150.0, key="v_rms", step=2.0)
        v_threshold = st.slider("Threshold (V)", 0.0, v_rms, key="v_threshold", step=1.0)
        frequency = st.slider("Frequency (Hz)", min_value=5.0, max_value=200.0, key="frequency", step=1.0)

        v_peak = v_rms * np.sqrt(2)
        st.caption(f"Peak ($V_p$): **{v_peak:.1f} V**")

        st.header("4. Busbars")
        col_add, col_rem = st.columns(2)
        if col_add.button("➕ Add"):
            st.session_state.busbars.append(
                {'side': 'Left', 'start_pos': 0.0, 'length': film_l_cm / 2.0, 'layer': 'Top ITO', 'signal': 'Positive (+)'}
            )
        if col_rem.button("🗑️ Remove") and len(st.session_state.busbars) > 1:
            st.session_state.busbars.pop()

        configured_busbars = []
        for i, bb in enumerate(st.session_state.busbars):
            current_signal = bb.get('signal', 'Positive (+)')
            current_layer = bb.get('layer', 'Top ITO')
            current_side = bb.get('side', 'Top')

            with st.expander(f"Bar #{i+1} ({current_layer})", expanded=False):
                side_options = ["Top", "Bottom", "Left", "Right"]
                side_idx = side_options.index(current_side) if current_side in side_options else 0
                side_key = f"side_{i}"
                if side_key not in st.session_state:
                    st.session_state[side_key] = side_options[side_idx]
                side = st.selectbox(f"Side #{i+1}", side_options, key=side_key)
                
                layer_options = ["Top ITO", "Bottom ITO"]
                layer_idx = layer_options.index(current_layer) if current_layer in layer_options else 0
                layer_key = f"lay_{i}"
                if layer_key not in st.session_state:
                    st.session_state[layer_key] = layer_options[layer_idx]
                layer = st.selectbox(f"Layer #{i+1}", layer_options, key=layer_key)

                max_edge_len = film_w_cm if side in ["Top", "Bottom"] else film_l_cm
                safe_start = min(float(bb.get('start_pos', 0.0)), max_edge_len - 0.1)
                max_allowed_len = max(0.1, max_edge_len - safe_start)
                safe_len = min(float(bb.get('length', max_edge_len)), max_allowed_len)

                start_key = f"start_{i}"
                if start_key not in st.session_state:
                    st.session_state[start_key] = safe_start
                start_pos = st.number_input(f"Start (cm) #{i+1}", 0.0, max(0.0, max_edge_len - 0.1), step=1.0, key=start_key)
                
                remaining_max_len = max(0.1, max_edge_len - start_pos)
                len_key = f"len_{i}"
                if len_key not in st.session_state:
                    st.session_state[len_key] = min(safe_len, remaining_max_len)
                length_val = st.number_input(f"Len (cm) #{i+1}", 0.1, remaining_max_len, step=1.0, key=len_key)
                
                sig_options = ["Positive (+)", "Negative (-)"]
                sig_idx = sig_options.index(current_signal) if current_signal in sig_options else 0
                sig_key = f"sig_{i}"
                if sig_key not in st.session_state:
                    st.session_state[sig_key] = sig_options[sig_idx]
                signal = st.selectbox(f"Signal #{i+1}", sig_options, key=sig_key)

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

        st.session_state.busbars = configured_busbars

        # --- EXECUTE SOLVER FOR SIDEBAR POWER CALCULATION ---
        _, _, _, _, active_p_w, _ = simulate_all_profiles(
            c_area, r_sq, busbar_tape_width_cm, busbar_sheet_res, film_w, film_l, configured_busbars, v_peak, v_rms, frequency, t_snapshot, resolution, resolution
        )

        st.divider()
        st.metric("Est. DC Supply Power Draw", f"{active_p_w:.3f} W")

# --- EXECUTE MAIN SOLVER ---
with st.spinner("Solving transient and steady-state fields..."):
    V_on, V_off, V_steady, total_steps, active_p_w, dc_power_w = simulate_all_profiles(
        c_area, r_sq, busbar_tape_width_cm, busbar_sheet_res, film_w, film_l, configured_busbars, v_peak, v_rms, frequency, t_snapshot, resolution, resolution
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

def get_threshold_colormap(base_colorscale_name):
    if v_threshold <= 0 or v_threshold >= v_rms:
        return base_colorscale_name
    
    frac = float(v_threshold / v_rms)
    if base_colorscale_name == "Inferno":
        return [
            [0.0, "#2b2f38"], [frac, "#2b2f38"],
            [frac + 1e-5, "#000004"], [0.4, "#57106e"], [0.7, "#bc3754"], [1.0, "#fcffa4"]
        ]
    elif base_colorscale_name == "Viridis":
        return [
            [0.0, "#2b2f38"], [frac, "#2b2f38"],
            [frac + 1e-5, "#440154"], [0.4, "#3b528b"], [0.7, "#21918c"], [1.0, "#fde725"]
        ]
    else: # Plasma
        return [
            [0.0, "#2b2f38"], [frac, "#2b2f38"],
            [frac + 1e-5, "#0d0887"], [0.4, "#6a00a8"], [0.7, "#b12a90"], [1.0, "#f0f921"]
        ]

def create_surface_figure(z_data, title_text, colorscale_name):
    custom_scale = get_threshold_colormap(colorscale_name)
    fig = go.Figure(data=[go.Surface(
        z=z_data, 
        x=x_grid_cm, 
        y=y_grid_cm, 
        colorscale=custom_scale,
        cmin=0.0,
        cmax=v_rms
    )])
    add_busbar_traces(fig)
    label_time = f"{t_snapshot_us} µs" if t_snapshot_us < 1000 else f"{t_snapshot_us/1000:.1f} ms"
    fig.update_layout(
        title=f"{title_text} ({label_time})",
        autosize=True, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='Width (cm)', 
            yaxis_title='Length (cm)', 
            zaxis_title='RMS Voltage (V)', 
            zaxis=dict(range=[0, v_rms * 1.1])
        )
    )
    return fig

# Row 1: Transient Maps Side-by-Side using native Streamlit bordered containers
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        min_on_val = np.min(V_on)
        st.metric("Min Voltage (Turn-ON)", f"{min_on_val:.2f} V")
        fig_on = create_surface_figure(V_on, "Turn-ON Differential Voltage RMS", "Inferno")
        st.plotly_chart(fig_on, use_container_width=True)

with col2:
    with st.container(border=True):
        max_off_val = np.max(V_off)
        st.metric("Max Voltage (Turn-OFF)", f"{max_off_val:.2f} V")
        fig_off = create_surface_figure(V_off, "Turn-OFF Differential Voltage RMS", "Viridis")
        st.plotly_chart(fig_off, use_container_width=True)

# --- CACHED TRANSIENT SWEEP STARTING AT t=0 UP TO DYNAMIC END TIME ---
@st.cache_data
def compute_transient_metrics_end_time(c_area_val, r_sq_val, w, l, bb_config, vp, v_rms_val, freq_val, nx, ny, t_end_ms):
    dx = w / (nx - 1)
    dy = l / (ny - 1)
    
    top_mask = np.zeros((ny, nx), dtype=bool)
    top_vals = np.zeros((ny, nx))
    bot_mask = np.zeros((ny, nx), dtype=bool)
    bot_vals = np.zeros((ny, nx))
    
    has_top, has_bot = False, False
    for bb in bb_config:
        ix_min = max(0, int(bb['x_min'] / w * (nx - 1)))
        ix_max = min(nx - 1, int(bb['x_max'] / w * (nx - 1)))
        iy_min = max(0, int(bb['y_min'] / l * (ny - 1)))
        iy_max = min(ny - 1, int(bb['y_max'] / l * (ny - 1)))
        val = (vp / 2.0) if bb['signal'] == 'Positive (+)' else (-vp / 2.0)
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

    dt = 0.2 * (r_sq_val * c_area_val) / (1/(dx**2) + 1/(dy**2))
    alpha_coeff = dt / (r_sq_val * c_area_val * dx**2)

    def run_steps(init_t, init_b, target_t, target_b, num_steps):
        V_t, V_b = init_t.copy(), init_b.copy()
        V_t[top_mask] = target_t[top_mask]
        V_b[bot_mask] = target_b[bot_mask]
        for _ in range(int(num_steps)):
            V_t_new = V_t.copy()
            V_t_new[1:-1, 1:-1] = V_t[1:-1, 1:-1] + alpha_coeff * (
                V_t[1:-1, 2:] + V_t[1:-1, :-2] + V_t[2:, 1:-1] + V_t[:-2, 1:-1] - 4*V_t[1:-1, 1:-1]
            )
            V_b_new = V_b.copy()
            V_b_new[1:-1, 1:-1] = V_b[1:-1, 1:-1] + alpha_coeff * (
                V_b[1:-1, 2:] + V_b[1:-1, :-2] + V_b[2:, 1:-1] + V_b[:-2, 1:-1] - 4*V_b[1:-1, 1:-1]
            )
            V_t_new[top_mask] = target_t[top_mask]
            V_b_new[bot_mask] = target_b[bot_mask]
            for V_arr in [V_t_new, V_b_new]:
                V_arr[0, :] = V_arr[1, :]
                V_arr[-1, :] = V_arr[-2, :]
                V_arr[:, 0] = V_arr[:, 1]
                V_arr[:, -1] = V_arr[:, -2]
            V_t, V_b = V_t_new, V_b_new
        return V_t, V_b

    time_us_list = np.linspace(0.0, t_end_ms * 1000.0, 40)
    min_on_vs_t = []
    max_off_vs_t = []

    steady_t, steady_b = run_steps(np.zeros((ny, nx)), np.zeros((ny, nx)), top_vals, bot_vals, int(5000e-6 / dt))

    for t_us in time_us_list:
        n_steps = max(0, int((t_us * 1e-6) / dt))
        if n_steps == 0:
            v_eff_on = np.abs(np.zeros((ny, nx)) - np.zeros((ny, nx))) / np.sqrt(2)
            v_eff_off = np.abs(steady_t - steady_b) / np.sqrt(2)
        else:
            # Turn-ON Min Voltage
            f_t_on, f_b_on = run_steps(np.zeros((ny, nx)), np.zeros((ny, nx)), top_vals, bot_vals, n_steps)
            v_eff_on = np.abs(f_t_on - f_b_on) / np.sqrt(2)
            
            # Turn-OFF Max Voltage (highest voltage remaining across film)
            f_t_off, f_b_off = run_steps(steady_t, steady_b, np.zeros((ny, nx)), np.zeros((ny, nx)), n_steps)
            v_eff_off = np.abs(f_t_off - f_b_off) / np.sqrt(2)
            
        min_on_vs_t.append(np.min(v_eff_on))
        max_off_vs_t.append(np.max(v_eff_off))
        
    return time_us_list, min_on_vs_t, max_off_vs_t

with st.spinner("Computing transient metric curves..."):
    t_axis, min_on_curve, max_off_curve = compute_transient_metrics_end_time(
        c_area, r_sq, film_w, film_l, configured_busbars, v_peak, v_rms, frequency, resolution, resolution,
        transient_end_ms
    )

# Row 2: Steady-State Map on the Left, 4th & 5th Transient Plots on the Right (using native bordered containers)
st.markdown("---")
row2_col1, row2_col2 = st.columns([1.2, 1.8])

with row2_col1:
    with st.container(border=True):
        st.subheader("Steady-State ON State Distribution")
        min_steady_val = np.min(V_steady)
        st.metric("Min Voltage (Steady-State)", f"{min_steady_val:.2f} V")
        fig_steady = create_surface_figure(V_steady, "Steady-State ON State Differential Voltage RMS", "Plasma")
        st.plotly_chart(fig_steady, use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.subheader(f"Transient Response Trajectories (0 to {transient_end_ms} ms)")
        
        # 4th Plot: Turn-ON Minimum Voltage vs Time (lowest voltage across film)
        fig_on_curve = go.Figure()
        fig_on_curve.add_trace(go.Scatter(
            x=t_axis, y=min_on_curve, mode='lines+markers', 
            name='Turn-ON Min Voltage', line=dict(color='firebrick', width=3)
        ))
        if v_threshold > 0:
            fig_on_curve.add_hline(y=v_threshold, line_dash="dash", line_color="orange", annotation_text=f"Threshold ({v_threshold}V)")
        fig_on_curve.update_layout(
            title="4. Turn-ON Lowest Voltage Across Film vs. Time",
            xaxis_title="Time (µs)",
            yaxis_title="Min RMS Voltage (V)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            height=250,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
        )
        st.plotly_chart(fig_on_curve, use_container_width=True)

        # 5th Plot: Turn-OFF Maximum Voltage vs Time (highest voltage across film)
        fig_off_curve = go.Figure()
        fig_off_curve.add_trace(go.Scatter(
            x=t_axis, y=max_off_curve, mode='lines+markers', 
            name='Turn-OFF Max Voltage', line=dict(color='royalblue', width=3)
        ))
        if v_threshold > 0:
            fig_off_curve.add_hline(y=v_threshold, line_dash="dash", line_color="orange", annotation_text=f"Threshold ({v_threshold}V)")
        fig_off_curve.update_layout(
            title="5. Turn-OFF Highest Voltage Across Film vs. Time",
            xaxis_title="Time (µs)",
            yaxis_title="Max RMS Voltage (V)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            height=250,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
        )
        st.plotly_chart(fig_off_curve, use_container_width=True)