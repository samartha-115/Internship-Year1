import math
import random


def normalize_angle(angle):
    """Normalize angle to be between -pi and pi."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def lines_intersect(p1, p2, p3, p4):
    """Check if line segment (p1, p2) intersects with (p3, p4)."""

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # Define 4 edges for line-of-sight intersection
        self.edges = [
            ((x, y), (x + width, y)),
            ((x + width, y), (x + width, y + height)),
            ((x + width, y + height), (x, y + height)),
            ((x, y + height), (x, y)),
        ]

    def intersects(self, p1, p2):
        for e in self.edges:
            if lines_intersect(p1, p2, e[0], e[1]):
                return True
        return False


class Target:
    def __init__(self, id, x, y, vx, vy, priority_class):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.target_vx = vx
        self.target_vy = vy
        self.speed = math.hypot(vx, vy)
        self.priority_class = priority_class  # 1 (high), 2 (medium), 3 (low)
        self.time_since_observed = random.uniform(0, 5)
        self.is_observed = False
        self.history = []

        # New features
        self.eliminated = False
        self.lock_acquired = False
        self.consecutive_obs_time = 0.0
        self.evasive_timer = random.uniform(3, 8)
        self.rcs_modifier = (
            0.3 if priority_class == 1 else (0.7 if priority_class == 2 else 1.2)
        )
        self.uncertainty = 0.0
        self.unpredictability = 0.0

    def move(self, dt, max_x, max_y, obstacles):
        if self.eliminated:
            return

        old_x, old_y = self.x, self.y
        old_vx, old_vy = self.vx, self.vy

        self.history.append((self.x, self.y))
        if len(self.history) > 20:
            self.history.pop(0)

        # Realistic movement: smoothly accelerate towards target velocity (curved flight paths)
        turn_rate = 25.0  # Speed at which targets adjust their vector

        if self.vx < self.target_vx:
            self.vx = min(self.vx + turn_rate * dt, self.target_vx)
        elif self.vx > self.target_vx:
            self.vx = max(self.vx - turn_rate * dt, self.target_vx)

        if self.vy < self.target_vy:
            self.vy = min(self.vy + turn_rate * dt, self.target_vy)
        elif self.vy > self.target_vy:
            self.vy = max(self.vy - turn_rate * dt, self.target_vy)

        # Calculate unpredictability (how fast the velocity vector angle is changing)
        if old_vx != 0 or old_vy != 0:
            old_angle = math.atan2(old_vy, old_vx)
            new_angle = math.atan2(self.vy, self.vx)
            angle_diff = abs(normalize_angle(new_angle - old_angle))
            self.unpredictability = angle_diff / dt

        # Evasive maneuvers (pick a new target vector)
        self.evasive_timer -= dt
        if self.evasive_timer <= 0:
            # Change target direction smoothly, maintain speed
            angle = math.atan2(self.target_vy, self.target_vx) + random.uniform(
                -math.pi / 1.2, math.pi / 1.2
            )
            self.target_vx = self.speed * math.cos(angle)
            self.target_vy = self.speed * math.sin(angle)
            self.evasive_timer = random.uniform(3, 8)

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off outer walls
        if self.x <= 0 or self.x >= max_x:
            self.vx *= -1
            self.target_vx *= -1
            self.x = max(0, min(self.x, max_x))
        if self.y <= 0 or self.y >= max_y:
            self.vy *= -1
            self.target_vy *= -1
            self.y = max(0, min(self.y, max_y))

        # Bounce off Obstacles
        for obs in obstacles:
            r = 6  # Target collision radius
            if (
                obs.x - r <= self.x <= obs.x + obs.width + r
                and obs.y - r <= self.y <= obs.y + obs.height + r
            ):
                self.x, self.y = old_x, old_y  # Step back out of the obstacle

                # Check which side was hit and reverse the corresponding velocity component
                if old_x < obs.x - r or old_x > obs.x + obs.width + r:
                    self.vx *= -1
                    self.target_vx *= -1
                if old_y < obs.y - r or old_y > obs.y + obs.height + r:
                    self.vy *= -1
                    self.target_vy *= -1

                # Apply reflected velocity immediately to prevent getting stuck
                self.x += self.vx * dt
                self.y += self.vy * dt

        # Update observation timers and uncertainty
        if not self.is_observed:
            self.time_since_observed += dt
            self.consecutive_obs_time = 0.0
            self.lock_acquired = False
            # Uncertainty grows based on velocity and time
            speed = math.hypot(self.vx, self.vy)
            self.uncertainty += speed * dt * 0.2
        else:
            self.time_since_observed = 0.0
            self.consecutive_obs_time += dt
            self.uncertainty = 0.0

            if self.consecutive_obs_time > 3.0:  # Lock acquired after 3 seconds
                self.lock_acquired = True

    def distance_to_radar(self, radar_x, radar_y):
        return math.hypot(self.x - radar_x, self.y - radar_y)


class RadarEnvironment:
    def __init__(self, num_targets=10, width=800, height=800):
        self.width = width
        self.height = height
        self.radar_x = width / 2
        self.radar_y = height / 2
        self.current_time = 0.0

        # Radar Physics
        self.radar_angle = 0.0  # Radians
        self.radar_slew_rate = math.pi  # 180 degrees per second
        self.beam_locked_on_id = None
        self.metrics = {"eliminated": 0, "avg_uncertainty": 0.0}

        self.targets = []
        self.target_id_counter = 0
        for _ in range(num_targets):
            self.spawn_target()

        # Obstacles
        self.obstacles = [Obstacle(150, 200, 100, 300), Obstacle(550, 400, 150, 80)]

    def spawn_target(self):
        x = random.choice([50, self.width - 50])
        y = random.choice([50, self.height - 50])
        vx = random.uniform(-40, 40)
        vy = random.uniform(-40, 40)
        p_class = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
        self.targets.append(Target(self.target_id_counter, x, y, vx, vy, p_class))
        self.target_id_counter += 1

    def is_line_of_sight_clear(self, target):
        p1 = (self.radar_x, self.radar_y)
        p2 = (target.x, target.y)
        for obs in self.obstacles:
            if obs.intersects(p1, p2):
                return False
        return True

    def step(self, dt, target_id_to_track):
        self.current_time += dt
        self.beam_locked_on_id = None

        # Determine angle to desired target
        desired_angle = self.radar_angle
        target_to_track = None
        for t in self.targets:
            if t.id == target_id_to_track and not t.eliminated:
                target_to_track = t
                break

        if target_to_track:
            dy = target_to_track.y - self.radar_y
            dx = target_to_track.x - self.radar_x
            desired_angle = math.atan2(dy, dx)

        # Rotate radar towards desired angle
        angle_diff = normalize_angle(desired_angle - self.radar_angle)
        max_rotation = self.radar_slew_rate * dt

        if abs(angle_diff) <= max_rotation:
            self.radar_angle = desired_angle
            if target_to_track and self.is_line_of_sight_clear(target_to_track):
                target_to_track.is_observed = True
                self.beam_locked_on_id = target_to_track.id
        else:
            self.radar_angle += max_rotation * (1 if angle_diff > 0 else -1)
            self.radar_angle = normalize_angle(self.radar_angle)
            if target_to_track:
                target_to_track.is_observed = False  # Still slewing, not observing yet!

        total_uncertainty = 0
        active_count = 0

        # Move targets and clean up eliminated
        for t in self.targets:
            if t.id != self.beam_locked_on_id:
                t.is_observed = False
            t.move(dt, self.width, self.height, self.obstacles)

            if not t.eliminated:
                total_uncertainty += t.uncertainty
                active_count += 1

        # Metrics
        if active_count > 0:
            self.metrics["avg_uncertainty"] = total_uncertainty / active_count

        # Only keep alive targets
        self.targets = [t for t in self.targets if not t.eliminated]

    def get_state(self):
        return self.targets
