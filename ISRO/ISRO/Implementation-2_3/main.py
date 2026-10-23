import csv
import math
import time
import tkinter as tk
from tkinter import messagebox, ttk

from math_3d import OrbitCamera3D, dist3, normalize_angle
from radar_env_3d import RadarEnvironment3D
from scheduler_3d import IntelligentScheduler3D, ReinforcementLearningTuner


class RadarSimulationApp3D:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Advanced 3D Intelligent Radar-Target Scheduling Simulation [Point 2 & 3]"
        )
        self.root.geometry("1560x980")
        self.root.configure(bg="#0b1120")

        # 3D Environment and AI (Expanded 2000m x 2000m x 800m Airspace)
        self.env = RadarEnvironment3D(width=2000.0, length=2000.0, height=800.0)
        self.scheduler = IntelligentScheduler3D()
        self.rl_tuner = ReinforcementLearningTuner(self.scheduler)

        # Simulation State
        self.dt = 0.08
        self.running = True
        self.after_id = None
        self.selected_target_id = None
        self.target_scores = {}
        self.target_telemetry = {}

        # 3D Camera Engine
        self.camera = OrbitCamera3D(
            target=(1000, 1000, 260), distance=2800.0, yaw=0.85, pitch=0.48, fov=780.0
        )
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_down_x = 0
        self.mouse_down_y = 0
        self.camera_follow_carrier = False

        # Visual toggles
        self.show_beam_cone = tk.BooleanVar(value=True)
        self.show_redundancy_grid = tk.BooleanVar(value=True)
        self.show_altitude_stalks = tk.BooleanVar(value=True)
        self.auto_tune_var = tk.BooleanVar(value=False)

        # Telemetry & History
        self.metrics_history = []
        self.csv_export_data = []
        self.start_time = time.time()
        self.frame_count = 0
        self.fire_hover = False
        self.salvo_hover = False

        # Precompute 3D points for ground distance rings
        c_ground = (1000.0, 1000.0, 0.0)
        self.ground_rings_3d = []
        for r_m in (400, 800, 1200, 1600):
            ring = []
            for seg in range(16):
                ang = (2 * math.pi * seg) / 16
                px = c_ground[0] + r_m * math.cos(ang)
                py = c_ground[1] + r_m * math.sin(ang)
                ring.append((px, py, 0.0))
            self.ground_rings_3d.append(ring)

        # Build GUI
        self.setup_ui()

        # Canvas Mouse Interaction for 3D Orbit, Direct Target Selection & Zoom
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # Start Simulation Loop
        self.update_loop()

    def setup_ui(self):
        # Configure TTK Styles with Dark Cyber-Tactical Theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 9)
        )
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 11, "bold"),
            background="#0f172a",
            foreground="#38bdf8",
        )
        style.configure(
            "Sub.TLabel",
            font=("Segoe UI", 9, "bold"),
            background="#0f172a",
            foreground="#94a3b8",
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            background="#1e293b",
            foreground="#f8fafc",
        )
        style.map("TButton", background=[("active", "#334155")])
        style.configure(
            "Red.TButton",
            font=("Segoe UI", 11, "bold"),
            background="#dc2626",
            foreground="white",
        )
        style.map("Red.TButton", background=[("active", "#ef4444")])
        style.configure(
            "Cyan.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#0284c7",
            foreground="white",
        )
        style.map("Cyan.TButton", background=[("active", "#0ea5e9")])

        # Main Layout: Left = 3D Tactical Canvas, Right = Control Panels
        self.main_frame = tk.Frame(self.root, bg="#0b1120")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Left Panel: 3D Tactical Airspace ---
        self.left_frame = tk.Frame(self.main_frame, bg="#0f172a", bd=1, relief=tk.SOLID)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Top Canvas HUD Bar
        hud_bar = tk.Frame(self.left_frame, bg="#1e293b", height=38)
        hud_bar.pack(fill=tk.X, side=tk.TOP)

        cam_lbl = tk.Label(
            hud_bar,
            text="3D VIEW:",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
        )
        cam_lbl.pack(side=tk.LEFT, padx=(10, 5))

        btn_orbit = tk.Button(
            hud_bar,
            text="Orbit View",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 8),
            command=self.set_view_orbit,
        )
        btn_orbit.pack(side=tk.LEFT, padx=3, pady=4)

        btn_top = tk.Button(
            hud_bar,
            text="Top C2 View",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 8),
            command=self.set_view_top,
        )
        btn_top.pack(side=tk.LEFT, padx=3, pady=4)

        btn_follow = tk.Button(
            hud_bar,
            text="Cockpit Follow",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 8),
            command=self.toggle_follow_carrier,
        )
        btn_follow.pack(side=tk.LEFT, padx=3, pady=4)

        # Checkboxes for 3D overlay layers
        tk.Checkbutton(
            hud_bar,
            text="3D Beam Cone",
            variable=self.show_beam_cone,
            bg="#1e293b",
            fg="#38bdf8",
            selectcolor="#0f172a",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            hud_bar,
            text="Redundancy Grid",
            variable=self.show_redundancy_grid,
            bg="#1e293b",
            fg="#f59e0b",
            selectcolor="#0f172a",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            hud_bar,
            text="Altitude Stalks",
            variable=self.show_altitude_stalks,
            bg="#1e293b",
            fg="#4ade80",
            selectcolor="#0f172a",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=8)

        self.hud_time_lbl = tk.Label(
            hud_bar,
            text="TIME: 0.0s",
            bg="#1e293b",
            fg="#f8fafc",
            font=("Consolas", 10, "bold"),
        )
        self.hud_time_lbl.pack(side=tk.RIGHT, padx=12)

        # 3D Canvas
        self.canvas = tk.Canvas(self.left_frame, bg="#070b14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Canvas Status & Priority Legend
        canvas_status = tk.Frame(self.left_frame, bg="#0f172a", height=24)
        canvas_status.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(
            canvas_status,
            text="Controls: Left-Drag: Orbit 3D | Right-Click: Select | Wheel: Zoom  ||  Threat Priority:  [● Class 1 High: Red]  [● Class 2 Med: Orange]  [● Class 3 Low: Yellow]",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 8, "bold"),
        ).pack(side=tk.LEFT, padx=10)

        # --- Right Panel: Multi-Tab Controls & Live Analytics ---
        self.right_frame = tk.Frame(
            self.main_frame, bg="#0f172a", width=620, bd=1, relief=tk.SOLID
        )
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        self.right_frame.pack_propagate(False)

        # Action Button Header with Dual Circular Red Fire Buttons and Tactical C2 Bay
        action_bar = tk.Frame(self.right_frame, bg="#0f172a")
        action_bar.pack(fill=tk.X, padx=8, pady=(4, 2))

        # 1. Left System Controls (Pause/Resume, CSV Export)
        left_ctrl = tk.Frame(
            action_bar, bg="#1e293b", bd=1, relief=tk.SOLID, padx=6, pady=4
        )
        left_ctrl.pack(side=tk.LEFT, padx=(0, 6), fill=tk.Y)

        self.btn_play = tk.Button(
            left_ctrl,
            text="⏸ PAUSE",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            width=8,
            command=self.toggle_play,
        )
        self.btn_play.pack(side=tk.TOP, pady=2)

        self.btn_export = tk.Button(
            left_ctrl,
            text="💾 CSV Export",
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            width=8,
            command=self.export_csv,
        )
        self.btn_export.pack(side=tk.TOP, pady=2)

        # 2. Fire Control Weapons Bay (Both Circular Red Buttons Side-by-Side)
        fire_bay = tk.Frame(
            action_bar, bg="#1e293b", bd=1, relief=tk.SOLID, padx=6, pady=2
        )
        fire_bay.pack(side=tk.LEFT, padx=(0, 6))

        # 2A. Single Fire Button (Circular Canvas, Red, 74x74)
        single_box = tk.Frame(fire_bay, bg="#1e293b")
        single_box.pack(side=tk.LEFT, padx=5)

        self.fire_canvas = tk.Canvas(
            single_box,
            width=74,
            height=74,
            bg="#1e293b",
            highlightthickness=0,
            cursor="hand2",
        )
        self.fire_canvas.pack(side=tk.TOP)
        self.fire_canvas.bind("<Button-1>", lambda e: self.fire_weapon())
        self.fire_canvas.bind("<Enter>", lambda e: self._on_fire_enter())
        self.fire_canvas.bind("<Leave>", lambda e: self._on_fire_leave())

        tk.Label(
            single_box,
            text="SINGLE FIRE",
            bg="#1e293b",
            fg="#f87171",
            font=("Segoe UI", 7, "bold"),
        ).pack(side=tk.TOP, pady=(1, 0))

        # 2B. Swarm Salvo Button (Circular Canvas, Red, 74x74 - Matching Fire Button!)
        salvo_box = tk.Frame(fire_bay, bg="#1e293b")
        salvo_box.pack(side=tk.LEFT, padx=5)

        self.salvo_canvas = tk.Canvas(
            salvo_box,
            width=74,
            height=74,
            bg="#1e293b",
            highlightthickness=0,
            cursor="hand2",
        )
        self.salvo_canvas.pack(side=tk.TOP)
        self.salvo_canvas.bind("<Button-1>", lambda e: self.fire_swarm_salvo())
        self.salvo_canvas.bind("<Enter>", lambda e: self._on_salvo_enter())
        self.salvo_canvas.bind("<Leave>", lambda e: self._on_salvo_leave())
        self.btn_salvo = self.salvo_canvas  # Backward compatibility & test asset

        tk.Label(
            salvo_box,
            text="SWARM SALVO",
            bg="#1e293b",
            fg="#f87171",
            font=("Segoe UI", 7, "bold"),
        ).pack(side=tk.TOP, pady=(1, 0))

        # 3. Weapons Fire Control Telemetry Card
        weap_status_frame = tk.Frame(
            action_bar, bg="#1e293b", bd=1, relief=tk.SOLID, padx=8, pady=4
        )
        weap_status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            weap_status_frame,
            text="⚡ WEAPONS FIRE CONTROL STATUS",
            bg="#1e293b",
            fg="#f87171",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        self.weap_status_lbl = tk.Label(
            weap_status_frame,
            text="Target: No Selection | Lock: NO LOCK (Gate: >=1.1s & <=12m)\nSwarm Salvo: 0 Locked Targets Ready",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Consolas", 8),
            justify=tk.LEFT,
        )
        self.weap_status_lbl.pack(anchor="w", pady=(3, 0))

        # Live Metrics HUD Card
        metrics_box = tk.Frame(self.right_frame, bg="#1e293b", bd=1, relief=tk.RIDGE)
        metrics_box.pack(fill=tk.X, padx=10, pady=(4, 2))

        self.metrics_label = tk.Label(
            metrics_box,
            text="3D Radar System Metrics Initializing...",
            bg="#1e293b",
            fg="#4ade80",
            font=("Consolas", 9, "bold"),
            justify=tk.LEFT,
            padx=10,
            pady=5,
        )
        self.metrics_label.pack(fill=tk.X)

        # Target Selector & Priority Class Switcher Card
        # Target Selection Status Card (Mouse-Driven Direct Selection)
        inspector_frame = tk.LabelFrame(
            self.right_frame,
            text="🎯 Target Selection & Priority Switcher (Mouse-Driven)",
            bg="#111827",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            bd=1,
            relief=tk.SOLID,
        )
        inspector_frame.pack(fill=tk.X, padx=10, pady=(2, 4))

        insp_top = tk.Frame(inspector_frame, bg="#111827")
        insp_top.pack(fill=tk.X, padx=6, pady=(3, 1))

        self.mouse_select_badge = tk.Label(
            insp_top,
            text="🖱️ MOUSE: Click any target in 3D View",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2,
            bd=1,
            relief=tk.SOLID,
        )
        self.mouse_select_badge.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_deselect = tk.Button(
            insp_top,
            text="✖ Deselect",
            bg="#334155",
            fg="#cbd5e1",
            font=("Segoe UI", 8, "bold"),
            command=self.clear_target_selection,
            padx=4,
            pady=1,
        )
        self.btn_deselect.pack(side=tk.LEFT, padx=(0, 4))

        # Backward compatibility object & var
        self.target_selector_var = tk.StringVar(value="-- Select Target --")
        self.target_combo = ttk.Combobox(
            self.root, textvariable=self.target_selector_var
        )  # Retained off-screen for compatibility

        self.target_details_lbl = tk.Label(
            insp_top,
            text="No target selected",
            bg="#111827",
            fg="#cbd5e1",
            font=("Consolas", 8),
        )
        self.target_details_lbl.pack(side=tk.LEFT, padx=4)

        # 3 Instant Priority Class Switcher Buttons
        btn_cls_frame = tk.Frame(inspector_frame, bg="#111827")
        btn_cls_frame.pack(fill=tk.X, padx=6, pady=(2, 4))

        self.btn_set_p1 = tk.Button(
            btn_cls_frame,
            text="🔴 Set Class 1 (High/Red)",
            bg="#7f1d1d",
            fg="#fecaca",
            activebackground="#dc2626",
            activeforeground="white",
            font=("Segoe UI", 8, "bold"),
            command=lambda: self.change_selected_target_class(1),
        )
        self.btn_set_p1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_set_p2 = tk.Button(
            btn_cls_frame,
            text="🟠 Set Class 2 (Med/Org)",
            bg="#7c2d12",
            fg="#fed7aa",
            activebackground="#ea580c",
            activeforeground="white",
            font=("Segoe UI", 8, "bold"),
            command=lambda: self.change_selected_target_class(2),
        )
        self.btn_set_p2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_set_p3 = tk.Button(
            btn_cls_frame,
            text="🟡 Set Class 3 (Low/Yel)",
            bg="#713f12",
            fg="#fef08a",
            activebackground="#ca8a04",
            activeforeground="white",
            font=("Segoe UI", 8, "bold"),
            command=lambda: self.change_selected_target_class(3),
        )
        self.btn_set_p3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Uncertainty & Redundancy Graph
        self.graph_canvas = tk.Canvas(
            self.right_frame,
            height=68,
            bg="#0a0f1d",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.graph_canvas.pack(fill=tk.X, padx=10, pady=(0, 4))

        # Tactical Controls Notebook (Tabs)
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        # Tab 1: AI Scheduling & 3D Redundancy
        tab_ai = tk.Frame(self.notebook, bg="#0f172a")
        self.notebook.add(tab_ai, text="🧠 3D AI Scheduler")
        self.setup_ai_tab(tab_ai)

        # Tab 2: Carrier Aircraft & Enemy Clusters
        tab_battlefield = tk.Frame(self.notebook, bg="#0f172a")
        self.notebook.add(tab_battlefield, text="✈️ Carrier & Swarms")
        self.setup_battlefield_tab(tab_battlefield)

        # Tab 3: Live 3D Target Telemetry Table
        tab_telemetry = tk.Frame(self.notebook, bg="#0f172a")
        self.notebook.add(tab_telemetry, text="📊 Target Telemetry")
        self.setup_telemetry_tab(tab_telemetry)

    def reset_scheduler_defaults(self):
        self.slider_dist.set(-4.5)
        self.slider_time.set(1.5)
        self.slider_class.set(18.0)
        self.slider_uncertainty.set(12.0)
        self.slider_slew.set(2.0)
        self.slider_redundancy.set(5.5)
        self.slider_cluster.set(8.0)
        self.slider_boost.set(45.0)

    def set_preset_intercept(self):
        self.slider_dist.set(-8.0)
        self.slider_time.set(0.5)
        self.slider_class.set(28.0)
        self.slider_uncertainty.set(18.0)
        self.slider_slew.set(1.0)
        self.slider_redundancy.set(2.0)
        self.slider_cluster.set(15.0)
        self.slider_boost.set(50.0)

    def set_preset_surveillance(self):
        self.slider_dist.set(-2.0)
        self.slider_time.set(4.0)
        self.slider_class.set(10.0)
        self.slider_uncertainty.set(6.0)
        self.slider_slew.set(4.0)
        self.slider_redundancy.set(12.0)
        self.slider_cluster.set(5.0)
        self.slider_boost.set(30.0)

    def setup_ai_tab(self, parent):
        container = tk.Frame(parent, bg="#0f172a")
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 1. RL Auto-Tune & Preset Controls Bar
        top_bar = tk.Frame(container, bg="#1e293b", bd=1, relief=tk.SOLID)
        top_bar.pack(fill=tk.X, pady=(0, 4), padx=2)

        tk.Checkbutton(
            top_bar,
            text="🤖 RL Auto-Tune",
            variable=self.auto_tune_var,
            bg="#1e293b",
            fg="#38bdf8",
            selectcolor="#0f172a",
            font=("Segoe UI", 8, "bold"),
        ).pack(side=tk.LEFT, padx=6, pady=3)

        preset_box = tk.Frame(top_bar, bg="#1e293b")
        preset_box.pack(side=tk.RIGHT, padx=4)

        tk.Label(
            preset_box,
            text="Presets:",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 7, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(
            preset_box,
            text="Defaults",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 7, "bold"),
            command=self.reset_scheduler_defaults,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            preset_box,
            text="Intercept",
            bg="#7f1d1d",
            fg="#fecaca",
            font=("Segoe UI", 7, "bold"),
            command=self.set_preset_intercept,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            preset_box,
            text="Surveillance",
            bg="#0369a1",
            fg="#e0f2fe",
            font=("Segoe UI", 7, "bold"),
            command=self.set_preset_surveillance,
        ).pack(side=tk.LEFT, padx=2)

        # 2. 2-Column Grid Frame for Weights
        weights_frame = tk.LabelFrame(
            container,
            text="⚖️ Multi-Criteria AI Scheduling Weights & Parameters",
            bg="#0f172a",
            fg="#38bdf8",
            font=("Segoe UI", 8, "bold"),
        )
        weights_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        weights_frame.columnconfigure(0, weight=1)
        weights_frame.columnconfigure(1, weight=1)

        def add_grid_slider(row, col, title, from_v, to_v, init_v, res=0.5):
            cell = tk.Frame(weights_frame, bg="#0f172a")
            cell.grid(row=row, column=col, sticky="ew", padx=6, pady=2)
            lbl = tk.Label(
                cell,
                text=title,
                bg="#0f172a",
                fg="#cbd5e1",
                font=("Segoe UI", 8, "bold"),
            )
            lbl.pack(anchor="w")
            s = tk.Scale(
                cell,
                from_=from_v,
                to=to_v,
                resolution=res,
                orient=tk.HORIZONTAL,
                bg="#1e293b",
                fg="#f8fafc",
                highlightthickness=0,
                bd=0,
                length=265,
            )
            s.set(init_v)
            s.pack(fill=tk.X, expand=True)
            return s

        # 4 Rows x 2 Columns Grid
        self.slider_dist = add_grid_slider(
            0, 0, "🎯 Range / Depth Penalty", -15.0, 0.0, self.scheduler.w_dist
        )
        self.slider_slew = add_grid_slider(
            0, 1, "⚡ 3D Beam Slew Penalty", 0.0, 20.0, self.scheduler.w_slew
        )

        self.slider_time = add_grid_slider(
            1, 0, "⏱️ Track Staleness Factor", 0.0, 10.0, self.scheduler.w_time
        )
        self.slider_redundancy = add_grid_slider(
            1, 1, "🔄 3D Redundancy Penalty", 0.0, 20.0, self.scheduler.w_redundancy
        )

        self.slider_class = add_grid_slider(
            2, 0, "⭐ Threat Priority Class Weight", 0.0, 35.0, self.scheduler.w_class
        )
        self.slider_cluster = add_grid_slider(
            2, 1, "🐝 Swarm Threat Multiplier", 0.0, 25.0, self.scheduler.w_cluster
        )

        self.slider_uncertainty = add_grid_slider(
            3,
            0,
            "📍 Spatial Uncertainty Urgency",
            0.0,
            30.0,
            self.scheduler.w_uncertainty,
        )
        self.slider_boost = add_grid_slider(
            3, 1, "🖱️ Manual Target Boost", 0.0, 80.0, 45.0, res=5.0
        )

    def setup_battlefield_tab(self, parent):
        container = tk.Frame(parent, bg="#0f172a")
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 1. Carrier Aircraft Controls (Point 3)
        c_frame = tk.LabelFrame(
            container,
            text="✈️ Carrier Aircraft Flight Dynamics (Point 3)",
            bg="#0f172a",
            fg="#38bdf8",
            font=("Segoe UI", 8, "bold"),
        )
        c_frame.pack(fill=tk.X, pady=(0, 4), padx=2)

        c_btn_frame = tk.Frame(c_frame, bg="#0f172a")
        c_btn_frame.pack(fill=tk.X, padx=4, pady=2)

        modes = [
            ("Orbit CAP", "ORBIT_CAP"),
            ("Evasive S-Turns", "EVASIVE_S_TURNS"),
            ("Straight Recon", "STRAIGHT_LEVEL"),
        ]
        for text, mode in modes:
            b = tk.Button(
                c_btn_frame,
                text=text,
                bg="#1e293b",
                fg="white",
                font=("Segoe UI", 8, "bold"),
                command=lambda m=mode: self.env.carrier.set_mode(m),
            )
            b.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.carrier_telemetry_lbl = tk.Label(
            c_frame,
            text="Carrier Position: (500, 500, 260) | Heading: 0° | Roll: 0°",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Consolas", 8),
            pady=2,
        )
        self.carrier_telemetry_lbl.pack(anchor="w", padx=6)

        # 2. Enemy Cluster Swarm Controls (Point 3)
        swarm_frame = tk.LabelFrame(
            container,
            text="🎯 Enemy Swarm Handling & Tactical Targeting (Point 3)",
            bg="#0f172a",
            fg="#f59e0b",
            font=("Segoe UI", 8, "bold"),
        )
        swarm_frame.pack(fill=tk.X, pady=(0, 4), padx=2)

        # Swarm Focus Row
        sw_focus_frame = tk.Frame(swarm_frame, bg="#0f172a")
        sw_focus_frame.pack(fill=tk.X, padx=4, pady=(2, 2))

        tk.Label(
            sw_focus_frame,
            text="Focus Swarm:",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 8, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 2))

        b_sw_a = tk.Button(
            sw_focus_frame,
            text="🔴 Swarm Alpha (Red)",
            bg="#7f1d1d",
            fg="#fecaca",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.focus_swarm("Alpha"),
        )
        b_sw_a.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        b_sw_b = tk.Button(
            sw_focus_frame,
            text="🟠 Swarm Beta (Org)",
            bg="#7c2d12",
            fg="#fed7aa",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.focus_swarm("Beta"),
        )
        b_sw_b.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        b_sw_g = tk.Button(
            sw_focus_frame,
            text="🟡 Swarm Gamma (Yel)",
            bg="#713f12",
            fg="#fef08a",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.focus_swarm("Gamma"),
        )
        b_sw_g.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # Tactical Maneuver Row
        sw_btn_frame = tk.Frame(swarm_frame, bg="#0f172a")
        sw_btn_frame.pack(fill=tk.X, padx=4, pady=2)

        btn_scatter_a = tk.Button(
            sw_btn_frame,
            text="⚡ Jink Alpha",
            bg="#1e293b",
            fg="#ff003b",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.env.trigger_cluster_scatter("Alpha"),
        )
        btn_scatter_a.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_scatter_b = tk.Button(
            sw_btn_frame,
            text="⚡ Jink All",
            bg="#1e293b",
            fg="#00f0ff",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.env.trigger_cluster_scatter("All"),
        )
        btn_scatter_b.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_salvo_tab = tk.Button(
            sw_btn_frame,
            text="🚀 Intercept Swarm",
            bg="#991b1b",
            fg="white",
            font=("Segoe UI", 7, "bold"),
            command=self.fire_swarm_salvo,
        )
        btn_salvo_tab.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_spawn = tk.Button(
            sw_btn_frame,
            text="+ Spawn Swarm",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 7, "bold"),
            command=lambda: self.env.spawn_cluster(),
        )
        btn_spawn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.cluster_info_lbl = tk.Label(
            swarm_frame,
            text="Swarms Active: Alpha (Red Strike), Beta (Org Interceptors), Gamma (Yel Drones) | 70 3D Trees (2.5x)",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 8),
            pady=2,
        )
        self.cluster_info_lbl.pack(anchor="w", padx=6)

        # 3. 3D Airspace & Terrain Specs Info (Point 2)
        geo_frame = tk.LabelFrame(
            container,
            text="📡 3D Airspace & Terrain Specs (Point 2)",
            bg="#0f172a",
            fg="#4ade80",
            font=("Segoe UI", 8, "bold"),
        )
        geo_frame.pack(fill=tk.X, padx=2, pady=(0, 2))

        geo_txt = (
            "• Airspace Volume: 2000m x 2000m x 800m  |  Dynamic Detection Depth: R_max 2600m\n"
            "• Terrain: Mt. Apex (460m), Mt. Sierra (340m), Titan Ridge (240m), Iron Mesa (150m) & 70 Evergreen Trees (2.5x)\n"
            "• Geometric Redundancy: 3D angular lattice tracks beam dwell coverage & overlap efficiency"
        )
        tk.Label(
            geo_frame,
            text=geo_txt,
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            padx=6,
            pady=2,
        ).pack(anchor="w")

    def setup_telemetry_tab(self, parent):
        container = tk.Frame(parent, bg="#0f172a")
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        columns = (
            "id",
            "cluster",
            "class",
            "range",
            "rcs",
            "uncert",
            "track",
            "score",
            "status",
            "lock",
        )
        self.tree = ttk.Treeview(container, columns=columns, show="headings", height=14)
        self.tree.heading("id", text="ID")
        self.tree.heading("cluster", text="Cluster")
        self.tree.heading("class", text="Class")
        self.tree.heading("range", text="Range (m)")
        self.tree.heading("rcs", text="RCS (m²)")
        self.tree.heading("uncert", text="Uncert (m)")
        self.tree.heading("track", text="Track Quality")
        self.tree.heading("score", text="Score")
        self.tree.heading("status", text="Status")
        self.tree.heading("lock", text="Lock (1.1s)")

        self.tree.column("id", width=36, anchor="center")
        self.tree.column("cluster", width=58, anchor="center")
        self.tree.column("class", width=42, anchor="center")
        self.tree.column("range", width=56, anchor="center")
        self.tree.column("rcs", width=48, anchor="center")
        self.tree.column("uncert", width=56, anchor="center")
        self.tree.column("track", width=68, anchor="center")
        self.tree.column("score", width=50, anchor="center")
        self.tree.column("status", width=72, anchor="center")
        self.tree.column("lock", width=64, anchor="center")

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_table_select)

    def on_table_select(self, event):
        if getattr(self, "_updating_tree", False):
            return
        selected = self.tree.selection()
        if selected:
            vals = self.tree.item(selected[0], "values")
            if vals and len(vals) > 0:
                try:
                    tid = int(str(vals[0]).replace("T", ""))
                    self.selected_target_id = tid
                    self.update_target_inspector()
                    self.draw_weapons_buttons()
                except (ValueError, IndexError):
                    pass

    # --- Mouse & Camera Controls ---
    def on_mouse_down(self, event):
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y
        self.mouse_down_x = event.x
        self.mouse_down_y = event.y

    def on_mouse_drag(self, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        # Orbit rotation
        self.camera.yaw += dx * 0.006
        self.camera.pitch += dy * 0.006
        self.camera.pitch = max(-1.42, min(1.42, self.camera.pitch))

    def on_mouse_up(self, event):
        # Distinguish direct click from orbit drag: drag < 6.0 pixels = direct target click
        dist_moved = math.hypot(
            event.x - self.mouse_down_x, event.y - self.mouse_down_y
        )
        if dist_moved < 6.0:
            self.pick_target_at_screen_pos(event.x, event.y)

    def on_mouse_wheel(self, event):
        # Zoom in/out across expanded airspace
        zoom_step = -event.delta * 1.5
        self.camera.distance = max(500.0, min(5500.0, self.camera.distance + zoom_step))

    def on_canvas_motion(self, event):
        # Dynamic cursor feedback: crosshair when hovering over clickable targets
        is_hovering_target = False
        for t in self.env.targets:
            if t.eliminated:
                continue
            u, v, _depth, vis = self.camera.project_point(t.position)
            if vis and math.hypot(u - event.x, v - event.y) <= 32.0:
                is_hovering_target = True
                break

        cursor_shape = "crosshair" if is_hovering_target else ""
        try:
            self.canvas.config(cursor=cursor_shape)
        except tk.TclError:
            pass

    def pick_target_at_screen_pos(self, sx, sy):
        """Picks the closest active 3D target near screen position (sx, sy) within 36px tolerance."""
        best_id = None
        min_screen_dist = 36.0
        for t in self.env.targets:
            if t.eliminated:
                continue
            u, v, _depth, vis = self.camera.project_point(t.position)
            if vis:
                d = math.hypot(u - sx, v - sy)
                if d < min_screen_dist:
                    min_screen_dist = d
                    best_id = t.id

        self.selected_target_id = best_id
        self.update_target_inspector()
        self.draw_weapons_buttons()

    def clear_target_selection(self):
        """Clears the currently selected target."""
        self.selected_target_id = None
        self.update_target_inspector()
        self.draw_weapons_buttons()

    def on_right_click(self, event):
        self.pick_target_at_screen_pos(event.x, event.y)

    def on_target_combo_select(self, event=None):
        val = self.target_selector_var.get()
        if val.startswith("T"):
            try:
                tid = int(val.split()[0].replace("T", ""))
                self.selected_target_id = tid
                self.update_target_inspector()
                self.draw_weapons_buttons()
            except (ValueError, IndexError):
                pass

    def change_selected_target_class(self, new_class):
        if self.selected_target_id is None:
            messagebox.showinfo(
                "No Target Selected",
                "Please click a target in the 3D tactical view first to select and change its priority class.",
            )
            return

        for t in self.env.targets:
            if t.id == self.selected_target_id:
                t.set_priority_class(new_class)
                self.update_target_inspector()
                self.update_telemetry_table()
                class_names = {
                    1: "Class 1 High Priority (Red)",
                    2: "Class 2 Medium Priority (Orange)",
                    3: "Class 3 Low Priority (Yellow)",
                }
                self.metrics_label.config(
                    text=f"UPDATED: Target T{t.id} set to {class_names.get(new_class, 'Unknown')} | RCS: {t.rcs}m²"
                )
                return

    def update_target_inspector(self):
        alive_targets = [
            (t.id, t.cluster_id, t.priority_class)
            for t in self.env.targets
            if not t.eliminated
        ]
        if (
            getattr(self, "target_combo", None) is not None
            and alive_targets != getattr(self, "_last_combo_targets", None)
        ):
            self._last_combo_targets = alive_targets
            options = [
                f"T{tid} - {cid or 'Solo'} (P{pcl})" for tid, cid, pcl in alive_targets
            ]
            self.target_combo["values"] = options

        if self.selected_target_id is None:
            self.target_selector_var.set("-- Select Target --")
            if hasattr(self, "mouse_select_badge"):
                self.mouse_select_badge.config(
                    text="🖱️ MOUSE: Click any target in 3D View",
                    bg="#1e293b",
                    fg="#38bdf8",
                )
            self.target_details_lbl.config(text="No target selected", fg="#94a3b8")
            return

        sel_t = None
        for t in self.env.targets:
            if t.id == self.selected_target_id and not t.eliminated:
                sel_t = t
                break

        if sel_t is None:
            self.selected_target_id = None
            self.target_selector_var.set("-- Select Target --")
            if hasattr(self, "mouse_select_badge"):
                self.mouse_select_badge.config(
                    text="🖱️ MOUSE: Click any target in 3D View",
                    bg="#1e293b",
                    fg="#38bdf8",
                )
            self.target_details_lbl.config(
                text="Target lost / eliminated", fg="#ef4444"
            )
            return

        expected_str = (
            f"T{sel_t.id} - {sel_t.cluster_id or 'Solo'} (P{sel_t.priority_class})"
        )
        if self.target_selector_var.get() != expected_str:
            self.target_selector_var.set(expected_str)

        lock_txt = (
            "LOCKED [READY]"
            if sel_t.lock_acquired
            else (
                f"LOCKING ({sel_t.consecutive_obs_time:.1f}s/1.1s)"
                if sel_t.is_observed
                else "NO LOCK"
            )
        )
        col = (
            "#ff003b"
            if sel_t.priority_class == 1
            else ("#ff7700" if sel_t.priority_class == 2 else "#ffee00")
        )
        if hasattr(self, "mouse_select_badge"):
            self.mouse_select_badge.config(
                text=f"🎯 SELECTED: T{sel_t.id} ({sel_t.cluster_id or 'Solo'})",
                bg="#0f172a",
                fg=col,
            )
        t_dist = dist3(self.env.carrier.position, sel_t.position)
        self.target_details_lbl.config(
            text=f"RCS: {sel_t.rcs}m² | Range: {t_dist:.0f}m | Uncert: {sel_t.uncertainty:.1f}m ({sel_t.track_state}) | {lock_txt}",
            fg=col,
        )

    def focus_swarm(self, cluster_id):
        target_id = self.env.target_swarm(cluster_id)
        if target_id is not None:
            self.selected_target_id = target_id
            self.update_target_inspector()
            self.draw_weapons_buttons()
            self.metrics_label.config(
                text=f"SWARM C2: Radar tracking Swarm {cluster_id} priority lead target T{target_id}"
            )
        else:
            messagebox.showinfo(
                "Swarm Eliminated",
                f"All targets in Swarm {cluster_id} have been neutralized.",
            )

    def _on_fire_enter(self):
        self.fire_hover = True
        self.draw_circular_fire_button()

    def _on_fire_leave(self):
        self.fire_hover = False
        self.draw_circular_fire_button()

    def _on_salvo_enter(self):
        self.salvo_hover = True
        self.draw_circular_salvo_button()

    def _on_salvo_leave(self):
        self.salvo_hover = False
        self.draw_circular_salvo_button()

    def draw_circular_fire_button(self):
        self.fire_canvas.delete("all")
        cx, cy = 37, 37

        # Check if selected target has lock acquired
        is_locked = False
        consec_t = 0.0
        if self.selected_target_id is not None:
            for t in self.env.targets:
                if t.id == self.selected_target_id:
                    is_locked = t.lock_acquired
                    consec_t = t.consecutive_obs_time
                    break

        # Red tactical styling depending on lock state
        if is_locked:
            ring_color = "#ff003b"
            glow_color = "#ef4444" if self.fire_hover else "#dc2626"
            btn_fill = "#b91c1c" if not self.fire_hover else "#dc2626"
            text_color = "#ffffff"
            sub_text = "LOCKED"
        elif consec_t > 0.0:
            ring_color = "#ea580c"
            glow_color = "#f97316"
            btn_fill = "#9a3412" if not self.fire_hover else "#c2410c"
            text_color = "#ffedd5"
            sub_text = f"{consec_t:.1f}s"
        else:
            ring_color = "#991b1b"
            glow_color = "#7f1d1d"
            btn_fill = "#450a0a" if not self.fire_hover else "#7f1d1d"
            text_color = "#fca5a5"
            sub_text = "1.1s"

        # Outer ring
        self.fire_canvas.create_oval(
            cx - 33, cy - 33, cx + 33, cy + 33, outline=ring_color, width=2.5
        )
        if is_locked or self.fire_hover:
            self.fire_canvas.create_oval(
                cx - 30,
                cy - 30,
                cx + 30,
                cy + 30,
                outline=glow_color,
                width=1.5,
                dash=(2, 2),
            )

        # Inner Button Circle
        self.fire_canvas.create_oval(
            cx - 26,
            cy - 26,
            cx + 26,
            cy + 26,
            fill=btn_fill,
            outline=ring_color,
            width=1.5,
        )

        # Center Text & Icon
        self.fire_canvas.create_text(
            cx, cy - 12, text="🔥", font=("Segoe UI Emoji", 10)
        )
        self.fire_canvas.create_text(
            cx, cy + 1, text="FIRE", fill=text_color, font=("Segoe UI", 8, "bold")
        )
        self.fire_canvas.create_text(
            cx, cy + 13, text=sub_text, fill=text_color, font=("Consolas", 7, "bold")
        )

    def draw_circular_salvo_button(self):
        self.salvo_canvas.delete("all")
        cx, cy = 37, 37

        # Check how many swarm targets are currently locked
        candidates = [
            t for t in self.env.targets if t.lock_acquired and not t.eliminated
        ]
        has_locks = len(candidates) > 0

        # Red tactical styling for Swarm Salvo
        if has_locks:
            ring_color = "#ff003b"
            glow_color = "#ef4444" if self.salvo_hover else "#dc2626"
            btn_fill = "#b91c1c" if not self.salvo_hover else "#dc2626"
            text_color = "#ffffff"
            sub_text = f"{len(candidates)} LOCK"
        else:
            ring_color = "#991b1b"
            glow_color = "#7f1d1d"
            btn_fill = "#450a0a" if not self.salvo_hover else "#7f1d1d"
            text_color = "#fca5a5"
            sub_text = "SALVO"

        # Outer ring
        self.salvo_canvas.create_oval(
            cx - 33, cy - 33, cx + 33, cy + 33, outline=ring_color, width=2.5
        )
        if has_locks or self.salvo_hover:
            self.salvo_canvas.create_oval(
                cx - 30,
                cy - 30,
                cx + 30,
                cy + 30,
                outline=glow_color,
                width=1.5,
                dash=(2, 2),
            )

        # Inner Button Circle
        self.salvo_canvas.create_oval(
            cx - 26,
            cy - 26,
            cx + 26,
            cy + 26,
            fill=btn_fill,
            outline=ring_color,
            width=1.5,
        )

        # Center Text & Icon
        self.salvo_canvas.create_text(
            cx, cy - 12, text="💥", font=("Segoe UI Emoji", 10)
        )
        self.salvo_canvas.create_text(
            cx, cy + 1, text="SALVO", fill=text_color, font=("Segoe UI", 8, "bold")
        )
        self.salvo_canvas.create_text(
            cx, cy + 13, text=sub_text, fill=text_color, font=("Consolas", 7, "bold")
        )

    def draw_weapons_buttons(self):
        self.draw_circular_fire_button()
        self.draw_circular_salvo_button()

    def set_view_orbit(self):
        self.camera_follow_carrier = False
        self.camera.target = [1000, 1000, 260]
        self.camera.distance = 2800.0
        self.camera.yaw = 0.85
        self.camera.pitch = 0.48

    def set_view_top(self):
        self.camera_follow_carrier = False
        self.camera.target = [1000, 1000, 260]
        self.camera.distance = 3200.0
        self.camera.yaw = 0.0
        self.camera.pitch = 1.42

    def toggle_follow_carrier(self):
        self.camera_follow_carrier = not self.camera_follow_carrier

    def toggle_play(self):
        self.running = not self.running
        self.btn_play.config(text="RESUME" if not self.running else "PAUSE")
        if not self.running and self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except (tk.TclError, ValueError):
                pass
            self.after_id = None
        elif self.running and self.after_id is None:
            self.update_loop()

    def fire_weapon(self):
        if self.selected_target_id is None:
            messagebox.showinfo(
                "No Target Selected",
                "Click a target in the 3D tactical view or telemetry table first to lock and fire.",
            )
            return

        for t in self.env.targets:
            if t.id == self.selected_target_id:
                if t.lock_acquired:
                    # NORMAL FIRE: Eliminates ONLY the single locked target
                    self.env.destroy_target(t.id)
                    self.selected_target_id = None
                    self.update_target_inspector()
                    self.draw_weapons_buttons()
                    messagebox.showinfo(
                        "Target Intercepted (Normal Fire)",
                        f"🎯 NORMAL PRECISION FIRE HIT!\n"
                        f"Target T{t.id} eliminated in 3D airspace.\n\n"
                        f"• Operational Mode: Single Target Precision Kill (Eliminates T{t.id} ONLY)\n"
                        f"• Tracking Dwell Under View: {t.consecutive_obs_time:.1f}s (Gate: >=1.1s)\n"
                        f"• Track Uncertainty: {t.uncertainty:.1f}m (Gate: <=12.0m)\n\n"
                        f"Tip: Use SWARM SALVO for Area-of-Effect elimination within 450m radius!",
                    )
                else:
                    messagebox.showwarning(
                        "Weapons Lock Required",
                        f"Target T{t.id} is not fully locked.\n\n"
                        f"Weapons system requires:\n"
                        f"• Continuous tracking under view >= 1.1s (Current: {t.consecutive_obs_time:.2f}s)\n"
                        f"• Position uncertainty <= 12.0m (Current: {t.uncertainty:.1f}m, State: {t.track_state})\n\n"
                        f"Keep radar beam pointing at target to acquire lock.",
                    )
                return

    def fire_swarm_salvo(self):
        # SWARM FIRE: Eliminates the locked main target AND all enemies within 450m radius
        destroyed = self.env.fire_swarm_salvo(
            self.selected_target_id, blast_radius=450.0
        )
        if destroyed:
            targets_str = ", ".join(f"T{tid}" for tid in destroyed)
            self.selected_target_id = None
            self.update_target_inspector()
            self.draw_weapons_buttons()
            messagebox.showinfo(
                "Swarm Salvo Successful",
                f"💥 SWARM SALVO AREA-OF-EFFECT INTERCEPT!\n\n"
                f"Operational Mode: Cluster Salvo (Locked Lead Target + 450m Blast Radius)\n"
                f"Neutralized {len(destroyed)} enemy targets:\n{targets_str}\n\n"
                f"• Blast Radius: 450.0m\n"
                f"• Reactive Scatter: Remaining swarm elements triggered evasive burst.",
            )
        else:
            if self.selected_target_id is None:
                messagebox.showwarning(
                    "No Target Selected for Salvo",
                    "Please click a target in the 3D tactical view first to designate the center of the Swarm Salvo.",
                )
            else:
                messagebox.showwarning(
                    "Weapons Lock Required for Salvo",
                    f"Selected target T{self.selected_target_id} does not meet the 1.1s lock gate.\n\n"
                    f"Ensure continuous tracking under view >= 1.1s and uncertainty <= 12.0m before firing salvo.",
                )

    def export_csv(self):
        filename = "radar_3d_metrics_export.csv"
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Time (s)",
                        "Carrier X",
                        "Carrier Y",
                        "Carrier Z",
                        "Carrier Heading (deg)",
                        "Target ID",
                        "Cluster",
                        "Class",
                        "Target X",
                        "Target Y",
                        "Target Z",
                        "Range (m)",
                        "RCS (m2)",
                        "Uncertainty (m)",
                        "Track State",
                        "AI Score",
                        "SNR (dB)",
                        "Pd",
                        "Status",
                        "Redundancy Factor",
                        "Lock Status",
                    ]
                )
                writer.writerows(self.csv_export_data)
            messagebox.showinfo(
                "Export Successful",
                f"Full 3D tactical data exported to {filename} ({len(self.csv_export_data)} records).",
            )
        except (OSError, csv.Error) as e:
            messagebox.showerror("Export Failed", str(e))

    # --- Master Update Loop ---
    def update_loop(self):
        if not self.running:
            return

        # 1. Update Camera viewport
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 50 and h > 50:
            self.camera.set_viewport(w, h)

        if self.camera_follow_carrier:
            self.camera.target = [
                self.env.carrier.x,
                self.env.carrier.y,
                self.env.carrier.z,
            ]

        # 2. Update AI Weights (Manual or RL Auto-Tune)
        if self.auto_tune_var.get():
            tuned = self.rl_tuner.update(self.dt, self.env)
            if tuned:
                s = self.scheduler
                self.slider_dist.set(s.w_dist)
                self.slider_time.set(s.w_time)
                self.slider_class.set(s.w_class)
                self.slider_uncertainty.set(s.w_uncertainty)
                self.slider_slew.set(s.w_slew)
                self.slider_redundancy.set(s.w_redundancy)
                self.slider_cluster.set(s.w_cluster)
        else:
            self.scheduler.w_dist = self.slider_dist.get()
            self.scheduler.w_time = self.slider_time.get()
            self.scheduler.w_class = self.slider_class.get()
            self.scheduler.w_uncertainty = self.slider_uncertainty.get()
            self.scheduler.w_slew = self.slider_slew.get()
            self.scheduler.w_redundancy = self.slider_redundancy.get()
            self.scheduler.w_cluster = self.slider_cluster.get()

        # 3. AI Scheduler Decision
        best_id, self.target_scores, self.target_telemetry = (
            self.scheduler.evaluate_targets(
                self.env, self.selected_target_id, self.slider_boost.get()
            )
        )

        # 4. Step 3D Physical Environment
        self.env.step(self.dt, best_id)

        # 5. Record Telemetry for CSV Export (every 1 second)
        self.frame_count += 1
        if self.frame_count % 12 == 0:
            t_now = round(self.env.current_time, 1)
            c = self.env.carrier
            c_head_deg = round(math.degrees(c.heading), 1)

            for t in self.env.targets:
                telem = self.target_telemetry.get(t.id, {})
                score = self.target_scores.get(t.id, -float("inf"))
                score_str = "BLOCKED" if score == -float("inf") else f"{score:.1f}"

                self.csv_export_data.append(
                    [
                        t_now,
                        round(c.x, 1),
                        round(c.y, 1),
                        round(c.z, 1),
                        c_head_deg,
                        f"T{t.id}",
                        t.cluster_id or "Solo",
                        t.priority_class,
                        round(t.x, 1),
                        round(t.y, 1),
                        round(t.z, 1),
                        round(telem.get("range", 0.0), 1),
                        t.rcs,
                        round(t.uncertainty, 1),
                        t.track_state,
                        score_str,
                        round(telem.get("snr", -99.0), 1),
                        round(telem.get("pd", 0.0), 2),
                        telem.get("status", "UNKNOWN"),
                        round(telem.get("redundancy", 0.0), 2),
                        "LOCKED" if t.lock_acquired else "NO",
                    ]
                )

            # Keep buffer bounded
            if len(self.csv_export_data) > 6000:
                self.csv_export_data = self.csv_export_data[-4000:]

            # Log to metrics history for graph
            self.metrics_history.append(
                (
                    t_now,
                    round(self.env.metrics["avg_uncertainty"], 2),
                    round(self.env.metrics["redundancy_efficiency"], 2),
                )
            )
            if len(self.metrics_history) > 60:
                self.metrics_history.pop(0)

        # 6. Render 3D Canvas
        self.render_3d()

        # Update circular fire and salvo buttons every 5 frames for smooth animation without overhead
        if self.frame_count % 5 == 0:
            self.draw_weapons_buttons()

        # 7. Update GUI Text, Inspector & Tables periodically (throttled for zero lag)
        if self.frame_count % 10 == 0:
            self.update_telemetry_table()
            self.update_target_inspector()
            self.draw_performance_graph()
            self.update_hud_text()

        # Next frame: 25 ms delay for silky smooth 40 FPS
        self.after_id = self.root.after(25, self.update_loop)

    def update_hud_text(self):
        self.hud_time_lbl.config(text=f"TIME: {self.env.current_time:.1f}s")
        m = self.env.metrics
        c = self.env.carrier

        rl_str = "ACTIVE (Self-Optimizing)" if self.auto_tune_var.get() else "MANUAL"
        text = (
            f"⚡ 3D RADAR AIRSPACE TELEMETRY | Mode: {c.mode} | AI RL Auto-Tune: {rl_str}\n"
            f"Targets Eliminated: {m['eliminated']} | Targets in Radar FOR: {m['active_in_for']}/{len(self.env.targets)} | "
            f"Weapons Locked: {m['locked_targets']}\n"
            f"Avg 3D Uncertainty: {m['avg_uncertainty']:.1f}m | Geometric Redundancy Efficiency: {m['redundancy_efficiency'] * 100:.1f}%"
        )
        self.metrics_label.config(text=text)

        c_info = (
            f"Carrier Pos: ({c.x:.0f}, {c.y:.0f}, {c.z:.0f}) | "
            f"Heading: {math.degrees(c.heading):.1f}° | Bank Roll: {math.degrees(c.roll):.1f}° | Pitch: {math.degrees(c.pitch):.1f}°"
        )
        self.carrier_telemetry_lbl.config(text=c_info)

        # Update Weapons Fire Control Status Card
        if hasattr(self, "weap_status_lbl"):
            sel_t = None
            if self.selected_target_id is not None:
                for t in self.env.targets:
                    if t.id == self.selected_target_id and not t.eliminated:
                        sel_t = t
                        break

            if sel_t:
                if sel_t.lock_acquired:
                    s_lock = "LOCKED [READY TO FIRE]"
                elif sel_t.is_observed:
                    s_lock = f"TRACKING {sel_t.consecutive_obs_time:.1f}s/1.1s (Uncert: {sel_t.uncertainty:.1f}m)"
                else:
                    s_lock = f"NOT TRACKED (Uncert: {sel_t.uncertainty:.1f}m)"
                single_txt = (
                    f"Normal Fire: T{sel_t.id} [P{sel_t.priority_class}] -> {s_lock} (Kills T{sel_t.id} ONLY)"
                )
                nearby_salvo = [
                    t
                    for t in self.env.targets
                    if not t.eliminated
                    and dist3(sel_t.position, t.position) <= 450.0
                ]
                salvo_txt = f"Swarm Salvo (450m AOE): {len(nearby_salvo)} Targets in Kill Radius (T{sel_t.id} + {len(nearby_salvo)-1} nearby)"
            else:
                single_txt = (
                    "Normal Fire: No Target Selected (Click target in 3D View to Select)"
                )
                locked_salvo = [
                    t for t in self.env.targets if t.lock_acquired and not t.eliminated
                ]
                salvo_txt = f"Swarm Salvo (450m AOE): {len(locked_salvo)} Locked Targets in Airspace"
            self.weap_status_lbl.config(text=f"{single_txt}\n{salvo_txt}")

    def draw_performance_graph(self):
        self.graph_canvas.delete("all")
        w = self.graph_canvas.winfo_width()
        h = self.graph_canvas.winfo_height()
        if w < 50 or h < 30 or len(self.metrics_history) < 2:
            return

        # Draw labels
        self.graph_canvas.create_text(
            8,
            10,
            anchor="w",
            text="3D Uncertainty (Green) & Redundancy Eff (Gold)",
            fill="#94a3b8",
            font=("Segoe UI", 7),
        )

        pts = self.metrics_history
        max_u = max(15.0, max(p[1] for p in pts))
        dx = w / max(1, len(pts) - 1)
        avail_h = max(10, h - 18)

        # Plot Uncertainty (Green line)
        for i in range(1, len(pts)):
            x1 = (i - 1) * dx
            y1 = h - (pts[i - 1][1] / max_u * avail_h) - 4
            x2 = i * dx
            y2 = h - (pts[i][1] / max_u * avail_h) - 4
            self.graph_canvas.create_line(x1, y1, x2, y2, fill="#4ade80", width=2)

        # Plot Redundancy Efficiency (Gold line)
        for i in range(1, len(pts)):
            x1 = (i - 1) * dx
            y1 = h - (max(0.0, min(1.0, pts[i - 1][2])) * avail_h) - 4
            x2 = i * dx
            y2 = h - (max(0.0, min(1.0, pts[i][2])) * avail_h) - 4
            self.graph_canvas.create_line(
                x1, y1, x2, y2, fill="#f59e0b", width=2, dash=(2, 2)
            )

    def update_telemetry_table(self):
        self._updating_tree = True
        try:
            sorted_targets = sorted(
                self.env.targets,
                key=lambda t: self.target_scores.get(t.id, -float("inf")),
                reverse=True,
            )

            existing_rows = self.tree.get_children()

            # Fast in-place value updates to eliminate DOM destruction & UI stutter
            if len(existing_rows) == len(sorted_targets):
                for row_id, t in zip(existing_rows, sorted_targets):
                    score = self.target_scores.get(t.id, -float("inf"))
                    score_str = "BLOCKED" if score == -float("inf") else f"{score:.1f}"
                    telem = self.target_telemetry.get(t.id, {})
                    range_val = f"{telem.get('range', 0.0):.0f}"
                    status = telem.get("status", "TRACKABLE")
                    uncert_val = f"{t.uncertainty:.1f}m"
                    track_val = t.track_state
                    lock_val = (
                        "LOCKED"
                        if t.lock_acquired
                        else (
                            f"{t.consecutive_obs_time:.1f}s" if t.is_observed else "NO"
                        )
                    )

                    self.tree.item(
                        row_id,
                        values=(
                            f"T{t.id}",
                            t.cluster_id or "Solo",
                            t.priority_class,
                            range_val,
                            f"{t.rcs:.2f}",
                            uncert_val,
                            track_val,
                            score_str,
                            status,
                            lock_val,
                        ),
                    )
                    if t.id == self.selected_target_id:
                        self.tree.selection_set(row_id)
            else:
                for row in existing_rows:
                    self.tree.delete(row)

                for t in sorted_targets:
                    score = self.target_scores.get(t.id, -float("inf"))
                    score_str = "BLOCKED" if score == -float("inf") else f"{score:.1f}"
                    telem = self.target_telemetry.get(t.id, {})
                    range_val = f"{telem.get('range', 0.0):.0f}"
                    status = telem.get("status", "TRACKABLE")
                    uncert_val = f"{t.uncertainty:.1f}m"
                    track_val = t.track_state
                    lock_val = (
                        "LOCKED"
                        if t.lock_acquired
                        else (
                            f"{t.consecutive_obs_time:.1f}s" if t.is_observed else "NO"
                        )
                    )

                    row_id = self.tree.insert(
                        "",
                        "end",
                        values=(
                            f"T{t.id}",
                            t.cluster_id or "Solo",
                            t.priority_class,
                            range_val,
                            f"{t.rcs:.2f}",
                            uncert_val,
                            track_val,
                            score_str,
                            status,
                            lock_val,
                        ),
                    )
                    if t.id == self.selected_target_id:
                        self.tree.selection_set(row_id)
        finally:
            self._updating_tree = False

    # --- 3D Projection & Rendering Engine ---
    def render_3d(self):
        self.canvas.delete("all")
        # Update camera frame basis once for ultra-fast point projections
        self.camera.update_frame()
        eye_pos = self.camera.get_eye_position()

        # Precomputed fast projection closure caching local registers
        ex, ey, ez = eye_pos
        fx, fy, fz = self.camera._fx, self.camera._fy, self.camera._fz
        rx, ry, rz = self.camera._rx, self.camera._ry, self.camera._rz
        ux, uy, uz = self.camera._ux, self.camera._uy, self.camera._uz
        fov = self.camera.fov
        hw, hh = self.camera._half_w, self.camera._half_h

        def fast_proj(p):
            vx = p[0] - ex
            vy = p[1] - ey
            vz = p[2] - ez
            cam_z = vx * fx + vy * fy + vz * fz
            if cam_z <= 20.0:
                return (0.0, 0.0, cam_z, False)
            scale = fov / cam_z
            cam_x = vx * rx + vy * ry + vz * rz
            cam_y = vx * ux + vy * uy + vz * uz
            return (hw + cam_x * scale, hh - cam_y * scale, cam_z, True)

        proj = fast_proj

        # 1. Draw 3D Ground Reference Grid (Z = 0)
        grid_color = "#132238"
        axis_color = "#1f3b60"
        ring_color = "#0f2d4a"

        # Grid lines along X and Y (Expanded 2000m x 2000m airspace)
        for gx in range(0, 2001, 400):
            p1 = (gx, 0, 0)
            p2 = (gx, 2000, 0)
            u1, v1, _, vis1 = proj(p1)
            u2, v2, _, vis2 = proj(p2)
            if vis1 and vis2:
                col = axis_color if gx in (0, 1000, 2000) else grid_color
                self.canvas.create_line(u1, v1, u2, v2, fill=col, width=1)

        for gy in range(0, 2001, 400):
            p1 = (0, gy, 0)
            p2 = (2000, gy, 0)
            u1, v1, _, vis1 = proj(p1)
            u2, v2, _, vis2 = proj(p2)
            if vis1 and vis2:
                col = axis_color if gy in (0, 1000, 2000) else grid_color
                self.canvas.create_line(u1, v1, u2, v2, fill=col, width=1)

        # Distance rings on ground around center (1000, 1000) using precomputed vertices
        for ring_pts in self.ground_rings_3d:
            pts = []
            for p in ring_pts:
                u, v, _, vis = proj(p)
                if vis:
                    pts.extend([u, v])
            if len(pts) >= 4:
                self.canvas.create_polygon(
                    pts, fill="", outline=ring_color, width=1, dash=(3, 3)
                )

        # 1.5 Draw 3D Ground Trees (2.5x Enlarged Realistic Pine Silhouettes with Viewport Culling)
        scr_w, scr_h = hw * 2 + 60, hh * 2 + 60
        for tree in self.env.trees:
            bu, bv, _, b_vis = proj((tree.x, tree.y, 0.0))
            if not b_vis or bu < -60 or bu > scr_w or bv < -60 or bv > scr_h:
                continue
            au, av, a_depth, a_vis = proj((tree.x, tree.y, tree.height))
            if not a_vis:
                continue

            scale = fov / max(20.0, a_depth)
            r_scr = max(4.0, min(55.0, tree.radius * scale))
            h_trunk = max(5.0, (bv - av) * 0.30)

            # Robust wood trunk
            self.canvas.create_line(bu, bv, bu, bv - h_trunk, fill="#3b2314", width=3)
            # Tiered evergreen pine canopy silhouette with realistic branch contours
            canopy_base_y = bv - h_trunk
            mid_y = canopy_base_y - (canopy_base_y - av) * 0.45
            self.canvas.create_polygon(
                [
                    bu - r_scr,
                    canopy_base_y,
                    bu + r_scr,
                    canopy_base_y,
                    bu + r_scr * 0.55,
                    mid_y,
                    au,
                    av,
                    bu - r_scr * 0.55,
                    mid_y,
                ],
                fill=tree.color,
                outline="#064e3b",
                width=1,
            )

        # 2. Draw 3D Terrain Mountain Massifs (Solid 3D Polyhedra with Backface Culling)
        for obs in self.env.obstacles:
            faces = obs.get_solid_faces()
            visible_faces = []
            for face_pts, normal, fill_c, outline_c in faces:
                # Backface culling: test if face normal points towards camera
                fc_x = (face_pts[0][0] + face_pts[2][0]) * 0.5
                fc_y = (face_pts[0][1] + face_pts[2][1]) * 0.5
                fc_z = (face_pts[0][2] + face_pts[2][2]) * 0.5
                to_eye = (eye_pos[0] - fc_x, eye_pos[1] - fc_y, eye_pos[2] - fc_z)
                if (
                    normal[0] * to_eye[0]
                    + normal[1] * to_eye[1]
                    + normal[2] * to_eye[2]
                    <= 0
                ):
                    continue
                f_dist = dist3((fc_x, fc_y, fc_z), eye_pos)
                visible_faces.append((f_dist, face_pts, fill_c, outline_c))

            visible_faces.sort(key=lambda x: x[0], reverse=True)

            for _, face_pts, fill_c, outline_c in visible_faces:
                proj_pts = [proj(p) for p in face_pts]
                if all(p[3] for p in proj_pts):
                    flat_coords = [c for p in proj_pts for c in p[:2]]
                    self.canvas.create_polygon(
                        flat_coords, fill=fill_c, outline=outline_c, width=1
                    )

            # Mountain Name tag
            center_top = (
                (obs.min_x + obs.max_x) / 2,
                (obs.min_y + obs.max_y) / 2,
                obs.max_z + 18,
            )
            cu, cv, _, cvis = proj(center_top)
            if cvis:
                self.canvas.create_text(
                    cu,
                    cv,
                    text=f"▲ {obs.name} (H:{obs.max_z:.0f}m)",
                    fill="#94a3b8",
                    font=("Segoe UI", 8, "bold"),
                )

        # 3. Draw 3D Geometric Redundancy Overlay (Point 2)
        if self.show_redundancy_grid.get():
            red_mgr = self.env.radar.redundancy_mgr
            # Draw angular search cells as projected cones/arcs from carrier
            c_pos = self.env.carrier.position
            for i in range(0, red_mgr.az_divs, 2):
                az = red_mgr.min_az + (i + 0.5) * red_mgr.cell_w_az
                w_az = normalize_angle(self.env.carrier.heading + az)
                # Cell line in 3D
                r_depth = 1400.0
                cell_pt = (
                    c_pos[0] + r_depth * math.cos(w_az),
                    c_pos[1] + r_depth * math.sin(w_az),
                    c_pos[2],
                )
                u1, v1, _, vis1 = proj(c_pos)
                u2, v2, _, vis2 = proj(cell_pt)
                if vis1 and vis2:
                    self.canvas.create_line(
                        u1, v1, u2, v2, fill="#1e3a5f", width=1, dash=(2, 4)
                    )

        # 4. Draw Carrier Aircraft (Ownship Radar Platform - Point 3)
        c = self.env.carrier
        c_pos = c.position
        cu, cv, _, c_vis = proj(c_pos)

        # Carrier Trail (Single polyline for maximum FPS)
        if len(c.history) > 1:
            c_trail = []
            for pt in c.history:
                u, v, _, vis = proj(pt)
                if vis:
                    c_trail.extend([u, v])
            if len(c_trail) >= 4:
                self.canvas.create_line(c_trail, fill="#0284c7", width=1.5, dash=(4, 4))

        # Carrier Altitude Stalk & Shadow on ground
        if self.show_altitude_stalks.get():
            c_ground = (c.x, c.y, 0)
            gu, gv, _, g_vis = proj(c_ground)
            if c_vis and g_vis:
                self.canvas.create_line(cu, cv, gu, gv, fill="#0284c7", dash=(2, 2))
                self.canvas.create_oval(
                    gu - 4, gv - 3, gu + 4, gv + 3, fill="#082f49", outline=""
                )

        if c_vis:
            # Draw Carrier Delta-Wing Aircraft
            head = c.heading
            roll = c.roll
            wing_span = 24.0
            nose_len = 30.0

            # Nose, Left wingtip, Right wingtip in 3D
            nose_3d = (
                c.x + nose_len * math.cos(head),
                c.y + nose_len * math.sin(head),
                c.z,
            )
            left_w_3d = (
                c.x - 12 * math.cos(head) - wing_span * math.sin(head),
                c.y - 12 * math.sin(head) + wing_span * math.cos(head),
                c.z - wing_span * math.sin(roll),
            )
            right_w_3d = (
                c.x - 12 * math.cos(head) + wing_span * math.sin(head),
                c.y - 12 * math.sin(head) - wing_span * math.cos(head),
                c.z + wing_span * math.sin(roll),
            )

            nu, nv, _, nvis = proj(nose_3d)
            lu, lv, _, lvis = proj(left_w_3d)
            ru, rv, _, rvis = proj(right_w_3d)

            if nvis and lvis and rvis:
                self.canvas.create_polygon(
                    [nu, nv, lu, lv, cu, cv, ru, rv],
                    fill="#0284c7",
                    outline="#38bdf8",
                    width=2,
                )

            self.canvas.create_text(
                cu,
                cv - 22,
                text="CARRIER (OWNSHIP)",
                fill="#38bdf8",
                font=("Consolas", 8, "bold"),
            )

        # 4.5 Active Continuous Radar Targeting Vector (Radar pointing/targeting a target at all times)
        active_target = None
        if self.env.actively_targeted_id is not None:
            for t in self.env.targets:
                if t.id == self.env.actively_targeted_id and not t.eliminated:
                    active_target = t
                    break

        if active_target is not None and c_vis:
            atu, atv, _, at_vis = proj(active_target.position)
            if at_vis:
                laser_color = "#00f0ff" if active_target.is_observed else "#38bdf8"
                laser_dash = () if active_target.lock_acquired else (4, 3)
                self.canvas.create_line(
                    cu,
                    cv,
                    atu,
                    atv,
                    fill=laser_color,
                    width=2 if active_target.lock_acquired else 1.5,
                    dash=laser_dash,
                )
                r_ret = 12
                self.canvas.create_oval(
                    atu - r_ret,
                    atv - r_ret,
                    atu + r_ret,
                    atv + r_ret,
                    outline=laser_color,
                    width=1.5,
                    dash=(2, 2),
                )
                self.canvas.create_text(
                    atu,
                    atv + 20,
                    text=f"🎯 RADAR TRACK: T{active_target.id}",
                    fill=laser_color,
                    font=("Consolas", 7, "bold"),
                )

        # 5. Draw 3D Volumetric Radar Beam Cone (Point 2)
        if self.show_beam_cone.get():
            apex, perimeter = self.env.radar.get_beam_cone_mesh(c_pos)
            apex_u, apex_v, _, apex_vis = proj(apex)

            beam_color = "#00f0ff" if self.env.beam_locked_on_id else "#334155"
            beam_dash = () if self.env.beam_locked_on_id else (4, 4)

            proj_perim = [proj(p) for p in perimeter]
            valid_pts = [p for p in proj_perim if p[3]]

            if apex_vis and len(valid_pts) >= 4:
                # Draw rays from apex to perimeter (cone boundaries)
                for i in range(0, len(valid_pts), 4):
                    pu, pv, _, _ = valid_pts[i]
                    self.canvas.create_line(
                        apex_u, apex_v, pu, pv, fill=beam_color, width=1, dash=beam_dash
                    )

                # Draw perimeter ellipse / ring
                poly_pts = [coord for pt in valid_pts for coord in pt[:2]]
                self.canvas.create_polygon(
                    poly_pts, fill="", outline=beam_color, width=1, dash=beam_dash
                )

        # 5.5 Swarm Salvo 450m Kill Zone Preview for selected target
        if self.selected_target_id is not None:
            sel_target = None
            for t in self.env.targets:
                if t.id == self.selected_target_id and not t.eliminated:
                    sel_target = t
                    break
            if sel_target is not None:
                ring_pts = []
                for seg in range(16):
                    ang = (2.0 * math.pi * seg) / 16.0
                    ring_pts.append(
                        (
                            sel_target.x + 450.0 * math.cos(ang),
                            sel_target.y + 450.0 * math.sin(ang),
                            sel_target.z,
                        )
                    )
                proj_ring = [proj(p) for p in ring_pts]
                if all(p[3] for p in proj_ring):
                    flat_ring = [coord for pt in proj_ring for coord in pt[:2]]
                    self.canvas.create_polygon(
                        flat_ring,
                        fill="",
                        outline="#f87171",
                        width=1.5,
                        dash=(5, 4),
                    )
                    ru, rv = proj_ring[0][0], proj_ring[0][1]
                    self.canvas.create_text(
                        ru,
                        rv - 8,
                        text="💥 SWARM SALVO RADIUS: 450m",
                        fill="#f87171",
                        font=("Consolas", 8, "bold"),
                    )

        # 6. Draw Enemy Targets & Clusters (Point 3)
        # Sort targets by depth from camera so near targets render over far targets
        draw_queue = []
        for t in self.env.targets:
            cam_d = dist3(t.position, self.camera.get_eye_position())
            draw_queue.append((cam_d, t))
        draw_queue.sort(key=lambda item: item[0], reverse=True)

        for _, t in draw_queue:
            tu, tv, depth, tvis = proj(t.position)
            if not tvis:
                continue

            # Priority Colors: Bright Red (Class 1), Bright Orange (Class 2), Bright Yellow (Class 3)
            p_color = t.color
            p_label = (
                "P1-RED"
                if t.priority_class == 1
                else ("P2-ORG" if t.priority_class == 2 else "P3-YEL")
            )

            # Ground stalk & Shadow
            if self.show_altitude_stalks.get():
                gu, gv, _, gvis = proj((t.x, t.y, 0))
                if gvis:
                    self.canvas.create_line(tu, tv, gu, gv, fill="#334155", dash=(2, 2))
                    self.canvas.create_oval(
                        gu - 3,
                        gv - 2,
                        gu + 3,
                        gv + 2,
                        fill="#0f172a",
                        outline="#1e293b",
                    )

            # 3D Uncertainty Bubble: compact and subtle so it never blocks the pilot's vision
            if t.uncertainty > 1.0:
                scale = self.camera.fov / max(20.0, depth)
                # Reduced radius (capped between 4 and 22 pixels)
                r_screen = min(22.0, max(4.0, t.uncertainty * scale * 0.25))
                self.canvas.create_oval(
                    tu - r_screen,
                    tv - r_screen,
                    tu + r_screen,
                    tv + r_screen,
                    outline=p_color,
                    width=1,
                    dash=(1, 4),
                )

            # Trail in 3D (Single polyline for maximum FPS)
            if len(t.history) > 1:
                t_trail = []
                for pt in t.history:
                    u, v, _, vis = proj(pt)
                    if vis:
                        t_trail.extend([u, v])
                if len(t_trail) >= 4:
                    self.canvas.create_line(t_trail, fill=p_color, width=1.2)

            # Target Icon (Bright Color mapped to Priority Class)
            icon_radius = 5.5
            self.canvas.create_oval(
                tu - icon_radius,
                tv - icon_radius,
                tu + icon_radius,
                tv + icon_radius,
                fill=p_color,
                outline="#ffffff",
                width=1.5,
            )

            # Active Radar Dwell Highlight
            if t.is_observed:
                self.canvas.create_oval(
                    tu - 12, tv - 12, tu + 12, tv + 12, outline="#00f0ff", width=2
                )
                if t.lock_acquired:
                    # Weapons lock crosshairs
                    self.canvas.create_line(
                        tu - 18, tv, tu + 18, tv, fill="#ff003b", width=2
                    )
                    self.canvas.create_line(
                        tu, tv - 18, tu, tv + 18, fill="#ff003b", width=2
                    )
                    self.canvas.create_text(
                        tu,
                        tv - 22,
                        text="LOCK",
                        fill="#ff003b",
                        font=("Consolas", 8, "bold"),
                    )

            # User Selection Highlight with Tactical Corner Brackets
            if t.id == self.selected_target_id:
                b_sz = 14
                self.canvas.create_line(
                    [
                        tu - b_sz,
                        tv - b_sz + 5,
                        tu - b_sz,
                        tv - b_sz,
                        tu - b_sz + 5,
                        tv - b_sz,
                    ],
                    fill="#38bdf8",
                    width=2,
                )
                self.canvas.create_line(
                    [
                        tu + b_sz - 5,
                        tv - b_sz,
                        tu + b_sz,
                        tv - b_sz,
                        tu + b_sz,
                        tv - b_sz + 5,
                    ],
                    fill="#38bdf8",
                    width=2,
                )
                self.canvas.create_line(
                    [
                        tu - b_sz,
                        tv + b_sz - 5,
                        tu - b_sz,
                        tv + b_sz,
                        tu - b_sz + 5,
                        tv + b_sz,
                    ],
                    fill="#38bdf8",
                    width=2,
                )
                self.canvas.create_line(
                    [
                        tu + b_sz - 5,
                        tv + b_sz,
                        tu + b_sz,
                        tv + b_sz,
                        tu + b_sz,
                        tv - b_sz + 5,
                    ],
                    fill="#38bdf8",
                    width=2,
                )
                self.canvas.create_text(
                    tu,
                    tv - 20,
                    text=f"[SELECTED T{t.id}]",
                    fill="#38bdf8",
                    font=("Consolas", 8, "bold"),
                )

            # Target telemetry label (Unified in single text call for maximum frame rate)
            self.canvas.create_text(
                tu + 12,
                tv,
                text=f"T{t.id} [{p_label}]\n{t.z:.0f}m",
                fill=p_color,
                font=("Consolas", 8, "bold"),
                anchor="w",
            )

        # 7. Draw 3D Explosions
        for exp in self.env.explosions:
            eu, ev, depth, evis = proj(exp["pos"])
            if evis:
                scale = self.camera.fov / max(20.0, depth)
                r = exp["radius"] * scale
                self.canvas.create_oval(
                    eu - r, ev - r, eu + r, ev + r, outline="#ef4444", width=2
                )

        # 7.5 Draw Swarm Salvo Blast Shockwaves
        if (
            hasattr(self.env, "last_blast_event")
            and self.env.last_blast_event is not None
        ):
            blast = self.env.last_blast_event
            elapsed = self.env.current_time - blast["time"]
            if elapsed < 1.0:
                cur_r = blast["radius"] * min(1.0, elapsed / 0.5)
                cx, cy, cz = blast["center"]
                blast_pts = []
                for seg in range(16):
                    ang = (2.0 * math.pi * seg) / 16.0
                    blast_pts.append(
                        (
                            cx + cur_r * math.cos(ang),
                            cy + cur_r * math.sin(ang),
                            cz,
                        )
                    )
                proj_blast = [proj(p) for p in blast_pts]
                if all(p[3] for p in proj_blast):
                    b_coords = [c for p in proj_blast for c in p[:2]]
                    self.canvas.create_polygon(
                        b_coords,
                        fill="",
                        outline="#ff003b",
                        width=2,
                        dash=(3, 3) if elapsed > 0.4 else (),
                    )


if __name__ == "__main__":
    root = tk.Tk()
    app = RadarSimulationApp3D(root)

    def on_close():
        app.running = False
        if hasattr(app, "after_id") and app.after_id is not None:
            try:
                root.after_cancel(app.after_id)
            except (tk.TclError, ValueError):
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
