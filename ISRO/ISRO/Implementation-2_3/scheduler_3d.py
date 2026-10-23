import math
import random

from math_3d import cartesian_to_spherical, normalize_angle


class IntelligentScheduler3D:
    """
    Advanced 3D Multi-Criteria Radar Resource Scheduler.
    Intelligently balances:
    - 3D Detection Depth & SNR Feasibility
    - 3D Beam Slew Rotation (Azimuth + Elevation)
    - 3D Search Airspace Geometric Redundancy (avoiding wasted overlapping dwells)
    - Carrier Aircraft Field of Regard (FOR) limits
    - Enemy Cluster Target Threat Densities
    - 3D Uncertainty and Target Maneuver Unpredictability
    - Reinforcement Learning Auto-Tuning
    """

    def __init__(
        self,
        weight_dist=-4.0,
        weight_time=2.5,
        weight_class=18.0,
        weight_slew=6.0,
        weight_redundancy=7.0,
        weight_cluster=8.0,
        weight_uncertainty=12.0,
    ):
        # Master scheduling weights
        self.w_dist = weight_dist
        self.w_time = weight_time
        self.w_class = weight_class
        self.w_slew = weight_slew
        self.w_redundancy = weight_redundancy
        self.w_cluster = weight_cluster
        self.w_uncertainty = weight_uncertainty

    def evaluate_targets(self, env, selected_target_id=None, selection_boost=40.0):
        """
        Evaluates all active targets in the 3D environment and selects the optimal
        target for the next radar beam dwell.
        Returns (best_target_id, target_scores, target_telemetry).
        """
        best_target = None
        best_score = -float("inf")
        target_scores = {}
        target_telemetry = {}

        carrier_pos = env.carrier.position
        beam_az = env.radar.beam_az
        beam_el = env.radar.beam_el
        sim_time = env.current_time

        for t in env.targets:
            # 1. 3D Range and Angles
            dx = t.x - carrier_pos[0]
            dy = t.y - carrier_pos[1]
            dz = t.z - carrier_pos[2]
            target_range, target_az, target_el = cartesian_to_spherical(dx, dy, dz)

            # 2. Check Carrier Aircraft Field of Regard (FOR)
            in_for, body_az, body_el, _r_body = env.carrier.is_in_field_of_regard(
                t.position
            )
            if not in_for:
                target_scores[t.id] = -float("inf")
                target_telemetry[t.id] = {
                    "range": target_range,
                    "az": target_az,
                    "el": target_el,
                    "status": "BLIND_ARC",
                    "snr": -99.0,
                    "pd": 0.0,
                    "redundancy": 0.0,
                    "uncertainty": t.uncertainty,
                    "track_state": t.track_state,
                }
                continue

            # 3. Check 3D Mountain Terrain Line of Sight (LOS)
            los_clear = env.is_line_of_sight_clear(t)
            if not los_clear:
                target_scores[t.id] = -float("inf")
                target_telemetry[t.id] = {
                    "range": target_range,
                    "az": target_az,
                    "el": target_el,
                    "status": "OCCLUDED",
                    "snr": -99.0,
                    "pd": 0.0,
                    "redundancy": 0.0,
                    "uncertainty": t.uncertainty,
                    "track_state": t.track_state,
                }
                continue

            # 4. 3D Detection Depth & SNR Feasibility with Spatial Uncertainty (Point 2)
            max_depth = env.radar.calculate_detection_depth(t.rcs)
            off_az = abs(normalize_angle(target_az - beam_az))
            off_el = abs(target_el - beam_el)
            off_boresight = math.hypot(off_az, off_el)

            snr_db = env.radar.calculate_snr(
                target_range, t.rcs, off_boresight, uncertainty_m=t.uncertainty
            )
            pd = env.radar.calculate_detection_probability(snr_db)

            if target_range > max_depth * 1.15:
                # Beyond radar detection depth envelope
                target_scores[t.id] = -float("inf")
                target_telemetry[t.id] = {
                    "range": target_range,
                    "az": target_az,
                    "el": target_el,
                    "status": "BEYOND_DEPTH",
                    "snr": snr_db,
                    "pd": pd,
                    "redundancy": 0.0,
                    "uncertainty": t.uncertainty,
                    "track_state": t.track_state,
                }
                continue

            # Normalized distance metric [0..1] scaled for expanded 2000m+ airspace
            norm_dist = target_range / 2400.0

            # Priority Threat Class (Class 1 = 3 pts, Class 2 = 2 pts, Class 3 = 1 pt)
            class_score = 4 - t.priority_class

            # Spatial Uncertainty Urgency: prioritizing targets before track divergence occurs
            uncertainty_urgency = (
                (t.uncertainty / 25.0) * (class_score**1.3) * self.w_uncertainty
            )

            # 5. 3D Beam Slew Penalty (Azimuth + Elevation rotation time)
            angular_dist = math.hypot(off_az, off_el)
            slew_penalty = angular_dist * self.w_slew

            # 6. 3D Search Airspace Geometric Redundancy Penalty (Point 2)
            # Penalizes dwelling in sectors that already have redundant coverage
            cell_redundancy = env.radar.redundancy_mgr.get_cell_redundancy(
                body_az, body_el, sim_time
            )
            redundancy_penalty = cell_redundancy * self.w_redundancy

            # 7. Enemy Cluster Threat Multiplier (Point 3)
            cluster_bonus = 0.0
            if t.cluster_id and t.cluster_id in env.clusters:
                cluster = env.clusters[t.cluster_id]
                alive_count = len(cluster.alive_targets)
                # Swarms of multiple coordinated targets pose escalated threat
                cluster_bonus = self.w_cluster * (alive_count / 4.0)

            # 8. 3D Maneuver / Unpredictability Bonus (Kalman-style tracking priority)
            maneuver_bonus = min(15.0, t.unpredictability * 6.0)

            # 9. Composite 3D Scheduling Score
            score = (
                (self.w_dist * norm_dist)
                + (self.w_time * t.time_since_observed)
                + (self.w_class * class_score)
                + uncertainty_urgency
                + cluster_bonus
                + maneuver_bonus
                - slew_penalty
                - redundancy_penalty
            )

            # Boost if user selected target
            if t.id == selected_target_id:
                score += selection_boost

            target_scores[t.id] = score
            target_telemetry[t.id] = {
                "range": target_range,
                "az": target_az,
                "el": target_el,
                "status": "TRACKABLE",
                "snr": snr_db,
                "pd": pd,
                "redundancy": cell_redundancy,
                "uncertainty": t.uncertainty,
                "track_state": t.track_state,
            }

            if score > best_score:
                best_score = score
                best_target = t.id

        return best_target, target_scores, target_telemetry


class ReinforcementLearningTuner:
    """
    Hill-Climbing Reinforcement Learning Engine for 3D Radar Scheduling.
    Continuously samples performance reward:
    Reward = - (Avg 3D Uncertainty) + (Redundancy Efficiency * 25) + (Locked Targets * 5)
    and autonomously optimizes scheduler weights.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.best_reward = -float("inf")
        self.best_weights = self.get_weights()
        self.timer = 0.0
        self.step_count = 0

    def get_weights(self):
        s = self.scheduler
        return (
            s.w_dist,
            s.w_time,
            s.w_class,
            s.w_slew,
            s.w_redundancy,
            s.w_cluster,
            s.w_uncertainty,
        )

    def set_weights(self, weights):
        s = self.scheduler
        (
            s.w_dist,
            s.w_time,
            s.w_class,
            s.w_slew,
            s.w_redundancy,
            s.w_cluster,
            s.w_uncertainty,
        ) = weights

    def update(self, dt, env):
        self.timer += dt
        if self.timer < 2.5:  # Evaluate every 2.5 seconds of simulation time
            return False

        self.timer = 0.0
        self.step_count += 1

        avg_u = env.metrics["avg_uncertainty"]
        eff = env.metrics["redundancy_efficiency"]
        locked = env.metrics["locked_targets"]

        # Reward function
        reward = -(avg_u * 0.5) + (eff * 35.0) + (locked * 8.0)

        if reward > self.best_reward:
            # Policy improvement: save new best weights
            self.best_reward = reward
            self.best_weights = self.get_weights()
        else:
            # Revert to best weights
            self.set_weights(self.best_weights)

        # Mutate weights with exploration noise
        current = list(self.get_weights())
        # Perturb one or two weights
        idx = random.randint(0, len(current) - 1)
        deltas = [-0.8, -0.4, 0.4, 0.8]
        current[idx] += random.choice(deltas)

        # Clamping
        current[0] = max(-15.0, min(0.0, current[0]))  # w_dist
        current[1] = max(0.0, min(12.0, current[1]))  # w_time
        current[2] = max(0.0, min(40.0, current[2]))  # w_class
        current[3] = max(0.0, min(20.0, current[3]))  # w_slew
        current[4] = max(0.0, min(20.0, current[4]))  # w_redundancy
        current[5] = max(0.0, min(25.0, current[5]))  # w_cluster
        current[6] = max(0.0, min(30.0, current[6]))  # w_uncertainty

        self.set_weights(current)
        return True
