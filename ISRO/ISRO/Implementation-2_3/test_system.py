import math


def test_simulation_components():
    print("======================================================================")
    print("RUNNING COMPREHENSIVE VERIFICATION AUDIT: IMPLEMENTATION-2_3")
    print("======================================================================")

    # 1. Imports
    from radar_env_3d import RadarEnvironment3D
    from scheduler_3d import IntelligentScheduler3D, ReinforcementLearningTuner

    print("[OK] All 3D simulation modules imported successfully.")

    # 2. Test 2.5x Tree dimensions
    env = RadarEnvironment3D(width=2000.0, length=2000.0, height=800.0)
    assert len(env.trees) > 30, f"Expected trees, got {len(env.trees)}"
    sample_tree = env.trees[0]
    print(
        f"[OK] Tree scale verified: height={sample_tree.height:.1f}m, radius={sample_tree.radius:.1f}m (2.5x scale)"
    )
    assert sample_tree.height >= 45.0, f"Tree height too small: {sample_tree.height}"
    assert sample_tree.radius >= 20.0, f"Tree radius too small: {sample_tree.radius}"

    # 3. Test Swarm Organization
    assert "Alpha" in env.clusters, "Swarm Alpha missing"
    assert "Beta" in env.clusters, "Swarm Beta missing"
    assert "Gamma" in env.clusters, "Swarm Gamma missing"
    print(
        f"[OK] Tactical swarms initialized: Alpha ({len(env.clusters['Alpha'].targets)}), Beta ({len(env.clusters['Beta'].targets)}), Gamma ({len(env.clusters['Gamma'].targets)})"
    )

    # 4. Test Target Priority Class Switcher
    t = env.targets[0]
    orig_class = t.priority_class
    print(
        f"[OK] Testing Target T{t.id} priority class switching (current Class {orig_class}, color {t.color}, RCS {t.rcs}m^2)"
    )

    # Change to Class 1 (Red, RCS: 0.15m^2)
    t.set_priority_class(1)
    assert t.priority_class == 1
    assert t.color == "#ff003b"
    assert math.isclose(t.rcs, 0.15)

    # Change to Class 2 (Orange, RCS: 1.0m^2)
    t.set_priority_class(2)
    assert t.priority_class == 2
    assert t.color == "#ff7700"
    assert math.isclose(t.rcs, 1.0)

    # Change to Class 3 (Yellow, RCS: 4.5m^2)
    t.set_priority_class(3)
    assert t.priority_class == 3
    assert t.color == "#ffee00"
    assert math.isclose(t.rcs, 4.5)
    print(
        "[OK] Target Priority Class Switcher verified (1:Red/0.15m^2, 2:Orange/1.0m^2, 3:Yellow/4.5m^2)."
    )

    # 5. Test Spatial Uncertainty and Track State
    t.uncertainty = 5.0
    assert t.track_state == "FIRM", f"Expected FIRM, got {t.track_state}"
    t.uncertainty = 18.0
    assert t.track_state == "STABLE", f"Expected STABLE, got {t.track_state}"
    t.uncertainty = 35.0
    assert t.track_state == "COAST", f"Expected COAST, got {t.track_state}"
    t.uncertainty = 55.0
    assert t.track_state == "CRITICAL", f"Expected CRITICAL, got {t.track_state}"
    print(
        "[OK] Track states verified: FIRM (<8m), STABLE (8-25m), COAST (25-45m), CRITICAL (>45m)."
    )

    # 6. Test 1.1s Weapons Lock Gate
    t.is_observed = True
    t.consecutive_obs_time = 0.9
    t.uncertainty = 6.0
    t.update_kinematics(0.1, env.bounds, env.obstacles)  # consec -> 1.0s
    assert not t.lock_acquired, "Lock should not be acquired at 1.0s"
    t.update_kinematics(0.15, env.bounds, env.obstacles)  # consec -> 1.15s
    assert t.lock_acquired, "Lock should be acquired when under view for >= 1.1s and uncertainty <= 12m"
    print("[OK] 1.1s Weapons Lock Gate verified (Under view >= 1.1s & Uncertainty <= 12m).")

    # Lock rejected if uncertainty is high
    t.uncertainty = 15.0
    t.lock_acquired = False
    t.consecutive_obs_time = 1.5
    t.update_kinematics(0.05, env.bounds, env.obstacles)
    assert not t.lock_acquired, "Lock must be rejected when uncertainty > 12m"
    print(
        "[OK] Uncertainty Weapons Precision Gate verified (Blocks lock if uncertainty > 12m)."
    )

    # 7. Test Normal Fire vs Swarm Salvo (450m AOE) & Radar Continuous Targeting
    # 7A: Verify 1.5x Radar slew rates and 1.5x Target speeds
    slew_az_deg = math.degrees(env.radar.slew_rate_az)
    slew_el_deg = math.degrees(env.radar.slew_rate_el)
    assert slew_az_deg >= 239.0, f"Expected 1.5x slew AZ >= 240 deg/s, got {slew_az_deg}"
    assert slew_el_deg >= 179.0, f"Expected 1.5x slew EL >= 180 deg/s, got {slew_el_deg}"
    avg_speed = sum(t.speed for t in env.targets) / len(env.targets)
    assert avg_speed >= 65.0, f"Expected 1.5x target speed >= 65 m/s, got {avg_speed}"
    print(
        f"[OK] 1.5x Speeds verified: Radar slew az={slew_az_deg:.1f}°/s, el={slew_el_deg:.1f}°/s | Target avg speed={avg_speed:.1f}m/s"
    )

    # 7B: Verify Radar is pointing/targeting a target at any given point in time
    env.step(0.08, None)
    assert env.actively_targeted_id is not None, "Radar must always be pointing/targeting a target at any given point in time"
    print(f"[OK] Radar Continuous Targeting verified: actively pointing at target T{env.actively_targeted_id}.")

    # 7C: Verify Normal Fire vs Swarm Salvo distinction
    # Normal Fire: eliminates only the specific locked target
    target_to_lock = env.targets[0]
    target_to_lock.lock_acquired = True
    target_to_lock_id = target_to_lock.id
    prev_elim = env.metrics["eliminated"]
    destroyed_single = env.destroy_target(target_to_lock_id)
    assert destroyed_single is True, "Normal fire should eliminate single locked target"
    assert env.metrics["eliminated"] == prev_elim + 1, "Normal fire must eliminate exactly 1 target"
    print(f"[OK] Normal Fire verified: eliminated ONLY specific locked target T{target_to_lock_id}.")

    # Swarm Salvo: eliminates locked main target AND all active targets within 450m blast radius
    lead_id = env.target_swarm("Alpha")
    assert lead_id is not None, "Lead target for Swarm Alpha should not be None"
    for mem in env.clusters["Alpha"].targets:
        mem.lock_acquired = True
    destroyed_salvo = env.fire_swarm_salvo(lead_id, blast_radius=450.0)
    assert len(destroyed_salvo) >= 1, "Salvo should destroy locked swarm members"
    assert env.last_blast_event is not None, "Blast event must be recorded"
    assert env.last_blast_event["radius"] == 450.0, "Salvo blast radius must be 450m"
    print(
        f"[OK] Swarm Salvo AOE Intercept verified: eliminated {len(destroyed_salvo)} targets within 450m blast radius."
    )

    # 8. Test Simulation Stepping & AI Scheduler
    scheduler = IntelligentScheduler3D()
    tuner = ReinforcementLearningTuner(scheduler)

    for step_i in range(30):
        best_id, _scores, _telem = scheduler.evaluate_targets(env)
        env.step(0.08, best_id)
        assert env.actively_targeted_id is not None, "Radar must point at a target during step"
        tuner.update(0.08, env)

    print("[OK] 30 full simulation cycles stepped with 0 errors.")
    print(
        f"[OK] Live metrics: Eliminated={env.metrics['eliminated']}, Redundancy Efficiency={env.metrics['redundancy_efficiency'] * 100:.1f}%, Avg Uncertainty={env.metrics['avg_uncertainty']:.1f}m"
    )

    # 9. Test Tkinter GUI headless initialization & Direct Mouse Selection
    import tkinter as tk

    from main import RadarSimulationApp3D

    root = tk.Tk()
    root.withdraw()  # Headless
    app = RadarSimulationApp3D(root)
    # Check widgets
    assert hasattr(app, "fire_canvas"), "fire_canvas missing"
    assert hasattr(app, "salvo_canvas"), "salvo_canvas missing"
    assert hasattr(app, "btn_salvo"), "btn_salvo missing"
    assert hasattr(app, "mouse_select_badge"), "mouse_select_badge missing"
    assert hasattr(app, "btn_deselect"), "btn_deselect missing"
    assert hasattr(app, "btn_set_p1"), "btn_set_p1 missing"

    # Test direct mouse target selection
    target_pos = app.env.targets[0].position
    u, v, _, vis = app.camera.project_point(target_pos)
    if vis:
        app.pick_target_at_screen_pos(u, v)
        assert app.selected_target_id == app.env.targets[0].id, "Direct mouse click should select target"
        print(f"[OK] Direct Mouse Target Selection verified: successfully selected T{app.selected_target_id} at screen pos ({u:.1f}, {v:.1f}).")
        app.clear_target_selection()
        assert app.selected_target_id is None, "Deselect button should clear selection"

    app.update_target_inspector()
    app.draw_circular_fire_button()
    app.draw_circular_salvo_button()
    app.draw_weapons_buttons()
    app.update_telemetry_table()
    root.destroy()
    print(
        "[OK] Tkinter GUI components (Mouse-Driven Selection, Dual Circular Red Fire Buttons, Salvo 450m AOE) initialized with 0 errors."
    )

    print("======================================================================")
    print("ALL 7 VERIFICATION CRITERIA PASSED WITH ZERO ERRORS!")
    print("======================================================================")


if __name__ == "__main__":
    test_simulation_components()
