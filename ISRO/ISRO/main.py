import csv
import math
import random
import time
import tkinter as tk
from tkinter import messagebox, ttk

from radar_env import RadarEnvironment
from scheduler import RuleBasedScheduler


class RadarSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Intelligent Radar-Target Scheduling Simulation")
        self.root.geometry("1450x950")
        self.root.configure(bg="#2b2b2b")

        # Main Layout
        self.main_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel (Simulation Canvas)
        self.sim_width = 800
        self.sim_height = 800
        self.canvas_frame = tk.Frame(
            self.main_frame, bg="black", bd=2, relief=tk.SUNKEN
        )
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.sim_width,
            height=self.sim_height,
            bg="#0f172a",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Panel (Controls & Stats)
        self.control_frame = tk.Frame(self.main_frame, bg="#2b2b2b", width=600)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.env = RadarEnvironment(
            num_targets=12, width=self.sim_width, height=self.sim_height
        )
        self.scheduler = RuleBasedScheduler()
        self.dt = 0.1
        self.running = True
        self.target_scores = {}
        self.selected_target_id = None

        # RL Auto-Tune Variables
        self.auto_tune_var = tk.BooleanVar(value=False)
        self.rl_timer = 0.0
        self.best_uncertainty = float("inf")
        self.best_weights = (
            self.scheduler.w_dist,
            self.scheduler.w_time,
            self.scheduler.w_class,
        )
        self.current_step = 0

        # Data export
        self.metrics_history = []
        self.csv_export_data = []
        self.start_time = time.time()

        self.setup_ui()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.update_loop()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TLabel", background="#2b2b2b", foreground="white", font=("Segoe UI", 10)
        )
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Red.TButton", font=("Segoe UI", 12, "bold"), foreground="red")
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 13, "bold"),
            background="#2b2b2b",
            foreground="#4da6ff",
        )

        # --- Controls Section ---
        ctrl_lbl = ttk.Label(
            self.control_frame,
            text="AI Scheduling Weights (Live)",
            style="Title.TLabel",
        )
        ctrl_lbl.pack(pady=(0, 5), anchor="w")

        ttk.Label(
            self.control_frame, text="Distance Weight (Closer = Higher Priority)"
        ).pack(anchor="w")
        self.dist_slider = tk.Scale(
            self.control_frame,
            from_=-20.0,
            to=0.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            length=450,
            bg="#2b2b2b",
            fg="white",
            highlightthickness=0,
        )
        self.dist_slider.set(self.scheduler.w_dist)
        self.dist_slider.pack(pady=(0, 2), anchor="w")

        ttk.Label(
            self.control_frame, text="Unobserved Time Weight (Longer = Higher Priority)"
        ).pack(anchor="w")
        self.time_slider = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=10.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            length=450,
            bg="#2b2b2b",
            fg="white",
            highlightthickness=0,
        )
        self.time_slider.set(self.scheduler.w_time)
        self.time_slider.pack(pady=(0, 2), anchor="w")

        ttk.Label(
            self.control_frame,
            text="Priority Class Weight (Class 1 = Highest Priority)",
        ).pack(anchor="w")
        self.class_slider = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=50.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            length=450,
            bg="#2b2b2b",
            fg="white",
            highlightthickness=0,
        )
        self.class_slider.set(self.scheduler.w_class)
        self.class_slider.pack(pady=(0, 2), anchor="w")

        ttk.Label(
            self.control_frame, text="User Selection Boost (Boost clicked target)"
        ).pack(anchor="w")
        self.boost_slider = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=100.0,
            resolution=5.0,
            orient=tk.HORIZONTAL,
            length=450,
            bg="#2b2b2b",
            fg="white",
            highlightthickness=0,
        )
        self.boost_slider.set(30.0)
        self.boost_slider.pack(pady=(0, 5), anchor="w")

        # AI Auto Tune Checkbox
        self.tune_check = tk.Checkbutton(
            self.control_frame,
            text="🤖 Enable AI Auto-Tuning (Reinforcement Learning)",
            variable=self.auto_tune_var,
            bg="#2b2b2b",
            fg="#4da6ff",
            selectcolor="#0f172a",
            font=("Segoe UI", 10, "bold"),
        )
        self.tune_check.pack(anchor="w", pady=(0, 10))

        btn_frame = tk.Frame(self.control_frame, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, pady=5)

        self.play_btn = ttk.Button(
            btn_frame, text="Pause", command=self.toggle_play, width=10
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.fire_btn = ttk.Button(
            btn_frame,
            text="🔥 FIRE! (Destroy Selected)",
            command=self.fire_weapon,
            style="Red.TButton",
        )
        self.fire_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.export_btn = ttk.Button(
            btn_frame, text="Export CSV", command=self.export_csv, width=12
        )
        self.export_btn.pack(side=tk.RIGHT, padx=5)

        # --- Metrics Dashboard ---
        metrics_lbl = ttk.Label(
            self.control_frame, text="Live Performance Metrics", style="Title.TLabel"
        )
        metrics_lbl.pack(pady=(10, 5), anchor="w")

        self.metrics_text = tk.StringVar()
        self.metrics_lbl = ttk.Label(
            self.control_frame,
            textvariable=self.metrics_text,
            font=("Consolas", 11),
            foreground="#22c55e",
        )
        self.metrics_lbl.pack(anchor="w", pady=(0, 5))

        # Graph Canvas
        self.graph = tk.Canvas(
            self.control_frame,
            width=450,
            height=100,
            bg="#1e293b",
            highlightthickness=0,
        )
        self.graph.pack(pady=(0, 10))

        # --- Stats Section ---
        stats_lbl = ttk.Label(
            self.control_frame, text="Live Target Statistics", style="Title.TLabel"
        )
        stats_lbl.pack(pady=(0, 5), anchor="w")

        columns = ("id", "class", "score", "los", "lock")
        self.tree = ttk.Treeview(
            self.control_frame, columns=columns, show="headings", height=10
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("class", text="Class")
        self.tree.heading("score", text="Score")
        self.tree.heading("los", text="Visibility")
        self.tree.heading("lock", text="Lock")

        self.tree.column("id", width=40, anchor="center")
        self.tree.column("class", width=60, anchor="center")
        self.tree.column("score", width=80, anchor="center")
        self.tree.column("los", width=80, anchor="center")
        self.tree.column("lock", width=80, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.selected_lbl = ttk.Label(
            self.control_frame, text="\nSelected Target: None", style="TLabel"
        )
        self.selected_lbl.pack(pady=5, anchor="w")

    def toggle_play(self):
        self.running = not self.running
        self.play_btn.config(text="Pause" if self.running else "Resume")
        if self.running:
            self.update_loop()

    def fire_weapon(self):
        if self.selected_target_id is not None:
            for t in self.env.targets:
                if t.id == self.selected_target_id:
                    if t.lock_acquired:
                        t.eliminated = True
                        self.env.metrics["eliminated"] += 1
                        self.env.spawn_target()
                        self.selected_target_id = None
                        self.selected_lbl.config(
                            text="\nSelected Target: None\nTARGET DESTROYED!"
                        )
                    else:
                        messagebox.showwarning(
                            "No Lock",
                            "Target must be continuously tracked for 3 seconds to acquire a weapons lock!",
                        )
                    return

    def export_csv(self):
        filename = "radar_metrics_export.csv"
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Time (s)",
                        "Total Targets Eliminated",
                        "Avg System Uncertainty",
                        "Target ID",
                        "Class",
                        "AI Score",
                        "Line of Sight",
                        "Lock Status",
                    ]
                )
                writer.writerows(self.csv_export_data)
            messagebox.showinfo(
                "Export Successful", f"Detailed data exported to {filename}"
            )
        except (OSError, csv.Error) as e:
            messagebox.showerror("Export Failed", str(e))

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        clicked_id = None
        for t in self.env.targets:
            if (t.x - x) ** 2 + (t.y - y) ** 2 < 625:
                clicked_id = t.id
                break
        self.selected_target_id = clicked_id
        if clicked_id is not None:
            self.selected_lbl.config(text=f"\nSelected Target: T{clicked_id}")
        else:
            self.selected_lbl.config(text="\nSelected Target: None")
        self.render()

    def rl_auto_tune(self):
        """Simple Hill-Climbing RL algorithm to minimize Avg System Uncertainty"""
        current_uncertainty = self.env.metrics["avg_uncertainty"]

        # If we improved, save the weights as the new best!
        if current_uncertainty < self.best_uncertainty:
            self.best_uncertainty = current_uncertainty
            self.best_weights = (
                self.scheduler.w_dist,
                self.scheduler.w_time,
                self.scheduler.w_class,
            )
        else:
            # We didn't improve, revert to best weights
            self.scheduler.w_dist, self.scheduler.w_time, self.scheduler.w_class = (
                self.best_weights
            )

        # Randomly mutate weights (Explore)
        self.scheduler.w_dist = max(
            -20.0, min(0.0, self.scheduler.w_dist + random.uniform(-1, 1))
        )
        self.scheduler.w_time = max(
            0.0, min(10.0, self.scheduler.w_time + random.uniform(-0.5, 0.5))
        )
        self.scheduler.w_class = max(
            0.0, min(50.0, self.scheduler.w_class + random.uniform(-2, 2))
        )

        # Update sliders to show the AI tweaking them
        self.dist_slider.set(self.scheduler.w_dist)
        self.time_slider.set(self.scheduler.w_time)
        self.class_slider.set(self.scheduler.w_class)

    def update_loop(self):
        if not self.running:
            return

        if self.auto_tune_var.get():
            self.rl_timer += self.dt
            if self.rl_timer >= 2.0:  # Check & Tweak every 2 seconds
                self.rl_auto_tune()
                self.rl_timer = 0.0
        else:
            # Manual override from sliders
            self.scheduler.w_dist = self.dist_slider.get()
            self.scheduler.w_time = self.time_slider.get()
            self.scheduler.w_class = self.class_slider.get()

        # AI evaluates all targets and decides which target to observe next
        observed_id, self.target_scores = self.scheduler.evaluate_targets(
            self.env, self.selected_target_id, self.boost_slider.get()
        )

        # Step the environment forward in time
        self.env.step(self.dt, observed_id)

        # Log metrics
        self.current_step += 1
        if self.current_step % 10 == 0:  # Every 1 sec (since dt=0.1)
            t = round(time.time() - self.start_time, 1)
            elim = self.env.metrics["eliminated"]
            avg_u = round(self.env.metrics["avg_uncertainty"], 2)

            # For the graph
            self.metrics_history.append((t, elim, avg_u))

            # For the detailed CSV export
            for target in self.env.targets:
                score = self.target_scores.get(target.id, 0)
                score_str = "HIDDEN" if score == -float("inf") else f"{score:.1f}"
                los = "YES" if self.env.is_line_of_sight_clear(target) else "NO"
                lock = "LOCKED" if target.lock_acquired else "NO"

                self.csv_export_data.append(
                    [
                        t,
                        elim,
                        avg_u,
                        f"T{target.id}",
                        target.priority_class,
                        score_str,
                        los,
                        lock,
                    ]
                )

        self.render()

        # Optimize performance: Only update heavy UI elements (table and graph) periodically
        if self.current_step % 5 == 0:
            self.update_table()
            self.draw_graph()

            # Update metrics text
            rl_text = (
                "[RL TUNING ACTIVE]" if self.auto_tune_var.get() else "[MANUAL CONTROL]"
            )
            self.metrics_text.set(
                f"{rl_text}\nTargets Destroyed: {self.env.metrics['eliminated']}\nAvg Uncertainty: {self.env.metrics['avg_uncertainty']:.1f}"
            )

        # Decrease delay for smoother canvas rendering (higher FPS)
        self.root.after(30, self.update_loop)

    def draw_graph(self):
        self.graph.delete("all")
        if len(self.metrics_history) < 2:
            return

        width = 450
        height = 100
        # Draw last 50 data points of uncertainty
        pts = self.metrics_history[-50:]
        max_u = max(10, max([p[2] for p in pts]))

        dx = width / 50
        for i in range(1, len(pts)):
            x1 = (i - 1) * dx
            y1 = height - (pts[i - 1][2] / max_u * height)
            x2 = i * dx
            y2 = height - (pts[i][2] / max_u * height)
            self.graph.create_line(x1, y1, x2, y2, fill="#22c55e", width=2)

    def update_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        sorted_targets = sorted(
            self.env.targets,
            key=lambda t: self.target_scores.get(t.id, -float("inf")),
            reverse=True,
        )

        for t in sorted_targets:
            score = self.target_scores.get(t.id, 0)
            score_str = "HIDDEN" if score == -float("inf") else f"{score:.1f}"
            los = "YES" if self.env.is_line_of_sight_clear(t) else "NO"
            lock = (
                "LOCKED"
                if t.lock_acquired
                else f"Acquiring ({t.consecutive_obs_time:.1f}s)"
            )

            item_id = self.tree.insert(
                "", "end", values=(f"T{t.id}", t.priority_class, score_str, los, lock)
            )
            if t.id == self.selected_target_id:
                self.tree.selection_set(item_id)

    def render(self):
        self.canvas.delete("all")
        rx, ry = self.env.radar_x, self.env.radar_y

        # Draw obstacles
        for obs in self.env.obstacles:
            self.canvas.create_rectangle(
                obs.x,
                obs.y,
                obs.x + obs.width,
                obs.y + obs.height,
                fill="#475569",
                outline="#334155",
                stipple="gray50",
            )

        # Draw rotating radar beam
        beam_len = 1200
        beam_end_x = rx + math.cos(self.env.radar_angle) * beam_len
        beam_end_y = ry + math.sin(self.env.radar_angle) * beam_len

        # Beam color changes if locked on
        beam_color = "#0ea5e9" if self.env.beam_locked_on_id is not None else "#334155"
        self.canvas.create_line(
            rx,
            ry,
            beam_end_x,
            beam_end_y,
            fill=beam_color,
            dash=(8, 4),
            width=2,
            tags="radar_beam",
        )

        # Draw radar at center
        self.canvas.create_oval(
            rx - 12,
            ry - 12,
            rx + 12,
            ry + 12,
            fill="#22c55e",
            outline="white",
            tags="radar",
        )
        self.canvas.create_text(
            rx,
            ry - 22,
            text="RADAR",
            fill="#22c55e",
            font=("Consolas", 10, "bold"),
            tags="radar",
        )

        # Time display
        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            text=f"Time: {self.env.current_time:.1f}s",
            fill="white",
            font=("Consolas", 14),
        )

        for t in self.env.targets:
            color = (
                "#ef4444"
                if t.priority_class == 1
                else ("#f97316" if t.priority_class == 2 else "#eab308")
            )

            # Uncertainty Bubble
            if t.uncertainty > 0:
                u_rad = min(t.uncertainty, 100)  # Cap visual size
                self.canvas.create_oval(
                    t.x - u_rad,
                    t.y - u_rad,
                    t.x + u_rad,
                    t.y + u_rad,
                    outline=color,
                    dash=(2, 2),
                )

            # Trail
            if len(t.history) > 1:
                for i in range(1, len(t.history)):
                    x1, y1 = t.history[i - 1]
                    x2, y2 = t.history[i]
                    self.canvas.create_line(
                        x1, y1, x2, y2, fill=color, stipple="gray25"
                    )

            # Observation highlight
            if t.is_observed:
                self.canvas.create_oval(
                    t.x - 15, t.y - 15, t.x + 15, t.y + 15, outline="#0ea5e9", width=2
                )
                if t.lock_acquired:
                    # Draw weapons lock crosshairs
                    self.canvas.create_line(
                        t.x - 20, t.y, t.x + 20, t.y, fill="red", width=2
                    )
                    self.canvas.create_line(
                        t.x, t.y - 20, t.x, t.y + 20, fill="red", width=2
                    )

            # Selection highlight
            if t.id == self.selected_target_id:
                self.canvas.create_oval(
                    t.x - 22,
                    t.y - 22,
                    t.x + 22,
                    t.y + 22,
                    outline="white",
                    width=2,
                    dash=(2, 2),
                )

            # Draw target dot and ID
            self.canvas.create_oval(t.x - 6, t.y - 6, t.x + 6, t.y + 6, fill=color)
            self.canvas.create_text(
                t.x + 18,
                t.y,
                text=f"T{t.id}",
                fill="white",
                font=("Consolas", 9, "bold"),
            )

            score = self.target_scores.get(t.id, -float("inf"))
            score_str = "HIDDEN" if score == -float("inf") else f"{score:.1f}"
            self.canvas.create_text(
                t.x, t.y - 18, text=score_str, fill="#94a3b8", font=("Consolas", 8)
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = RadarSimulationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
    root.mainloop()
