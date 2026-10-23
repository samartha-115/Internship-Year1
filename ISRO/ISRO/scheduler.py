import math
import random

from radar_env import normalize_angle


class RuleBasedScheduler:
    def __init__(self, weight_dist=-5.0, weight_time=2.0, weight_class=20.0):
        # Weights for the scoring function
        self.w_dist = weight_dist
        self.w_time = weight_time
        self.w_class = weight_class

    def evaluate_targets(self, env, selected_target_id=None, selection_boost=0.0):
        best_target = None
        best_score = -float("inf")
        target_scores = {}

        radar_x = env.radar_x
        radar_y = env.radar_y
        radar_angle = env.radar_angle
        targets = env.targets

        for t in targets:
            # 1. Line of sight check
            if not env.is_line_of_sight_clear(t):
                target_scores[t.id] = -float("inf")
                continue

            dist = t.distance_to_radar(radar_x, radar_y)

            # 2. RCS Probabilistic Detection
            # Stealth targets (Class 1) have lower rcs_modifier
            prob = 1.0 - (dist / 1400.0) * (1.0 / t.rcs_modifier)
            prob = max(0.05, min(1.0, prob))  # Minimum 5% chance to see

            # Only roll if the target is NOT currently being actively tracked
            if not t.is_observed and random.random() > prob:
                target_scores[t.id] = -float("inf")  # Radar failed to ping it!
                continue

            norm_dist = (dist / t.rcs_modifier) / 1000.0
            class_score = 4 - t.priority_class

            # 3. Beam Slewing Penalty
            dy = t.y - radar_y
            dx = t.x - radar_x
            target_angle = math.atan2(dy, dx)
            angle_diff = abs(normalize_angle(target_angle - radar_angle))
            angle_penalty = angle_diff * 5.0

            # 4. Predictability Penalty (Kalman-style logic)
            # If target flies straight (unpredictability ~ 0), penalize it 10 points.
            # If maneuvering sharply, penalty is 0.
            predictability_penalty = max(0.0, 10.0 - (t.unpredictability * 5.0))

            # Score formula
            score = (
                (self.w_dist * norm_dist)
                + (self.w_time * t.time_since_observed)
                + (self.w_class * class_score)
                - angle_penalty
                - predictability_penalty
            )

            # Boost score if this target was manually selected by the user
            if t.id == selected_target_id:
                score += selection_boost

            target_scores[t.id] = score

            if score > best_score:
                best_score = score
                best_target = t.id

        return best_target, target_scores
