import time
import tkinter as tk

from main import RadarSimulationApp3D


def profile_fps():
    print("======================================================================")
    print("PROFILING 3D RADAR SIMULATION FPS & SMOOTHNESS")
    print("======================================================================")

    root = tk.Tk()
    root.withdraw()
    app = RadarSimulationApp3D(root)

    # Simulate window size
    app.canvas.winfo_width = lambda: 940
    app.canvas.winfo_height = lambda: 880
    app.graph_canvas.winfo_width = lambda: 600
    app.graph_canvas.winfo_height = lambda: 68

    # Warm-up frames
    for _ in range(20):
        best_id, app.target_scores, app.target_telemetry = (
            app.scheduler.evaluate_targets(app.env)
        )
        app.env.step(app.dt, best_id)
        app.render_3d()
        app.draw_circular_fire_button()
        app.update_telemetry_table()
        app.update_target_inspector()
        app.draw_performance_graph()
        root.update_idletasks()

    # Benchmark 200 simulation & render frames
    N = 200
    start = time.perf_counter()
    for f in range(N):
        best_id, app.target_scores, app.target_telemetry = (
            app.scheduler.evaluate_targets(app.env)
        )
        app.env.step(app.dt, best_id)
        app.render_3d()
        if f % 5 == 0:
            app.draw_weapons_buttons()
        if f % 10 == 0:
            app.update_telemetry_table()
            app.update_target_inspector()
            app.draw_performance_graph()
            app.update_hud_text()
        root.update_idletasks()

    elapsed = time.perf_counter() - start
    avg_frame_time_ms = (elapsed / N) * 1000.0
    effective_fps = N / elapsed

    root.destroy()

    print(f"Total benchmark frames: {N}")
    print(f"Total elapsed time:     {elapsed:.3f} s")
    print(f"Average frame latency:  {avg_frame_time_ms:.2f} ms")
    print(f"Effective Frame Rate:   {effective_fps:.1f} FPS")
    print("======================================================================")

    assert effective_fps >= 15.0, (
        f"Performance issue: {effective_fps:.1f} FPS is below smooth target"
    )
    print(f"[PASS] Smooth execution confirmed ({effective_fps:.1f} FPS achieved).")


if __name__ == "__main__":
    profile_fps()
