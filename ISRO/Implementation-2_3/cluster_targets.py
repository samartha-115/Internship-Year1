import math
import random

from math_3d import (
    add3,
    dist3,
    mul3,
    norm3,
    normalize3,
    sub3,
)


class Target3D:
    """
    Individual 3D Airborne Target with realistic kinematics, 3D curved trajectories,
    RCS profile, and unpredictability calculation.
    """

    def __init__(
        self,
        id,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        priority_class=2,
        cluster_id=None,
        cluster_color="#00e5ff",
    ):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.target_vx = vx
        self.target_vy = vy
        self.target_vz = vz

        self.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        self.priority_class = priority_class  # 1 (Critical), 2 (Medium), 3 (Low)
        self.cluster_id = cluster_id  # None or cluster ID tag (e.g., 'Alpha', 'Beta')

        # Bright, clearly differentiable Priority Colors:
        # Red = Class 1 (Highest), Orange = Class 2 (Medium), Yellow = Class 3 (Low)
        if priority_class == 1:
            self.color = "#ff003b"  # Bright Electric Crimson Red (Highest priority)
        elif priority_class == 2:
            self.color = "#ff7700"  # Bright Vivid Tangerine Orange (Medium priority)
        else:
            self.color = "#ffee00"  # Bright Pure Glowing Yellow (Low priority)
        self.cluster_color = self.color

        # Radar Cross Section (RCS) in m^2
        if priority_class == 1:
            self.rcs = 0.15  # Stealth / Low Observability
        elif priority_class == 2:
            self.rcs = 1.0  # Medium Drone / Cruise Missile
        else:
            self.rcs = 4.5  # Conventional Fighter / Transport

        # Tracking state
        self.time_since_observed = random.uniform(0.5, 4.0)
        self.is_observed = False
        self.consecutive_obs_time = 0.0
        self.lock_acquired = False
        self.eliminated = False

        # 3D Uncertainty & Predictability metrics
        self.uncertainty = 0.0  # 3D position uncertainty radius (meters)
        self.unpredictability = 0.0  # 3D angular rate of direction change (rad/s)
        self.evasive_timer = random.uniform(2.5, 6.0)

        # 3D Trail history
        self.history = []

    @property
    def position(self):
        return (self.x, self.y, self.z)

    @property
    def velocity(self):
        return (self.vx, self.vy, self.vz)

    def update_kinematics(self, dt, airspace_bounds, obstacles):
        if self.eliminated:
            return

        min_x, max_x, min_y, max_y, min_z, max_z = airspace_bounds
        old_v = (self.vx, self.vy, self.vz)

        # Record trail
        self.history.append((self.x, self.y, self.z))
        if len(self.history) > 14:
            self.history.pop(0)

        # Smooth acceleration towards target velocity (curved 3D flight paths)
        turn_rate = 48.0  # m/s^2 acceleration limit (1.5x speed)
        for i, (v, tv) in enumerate(
            [
                (self.vx, self.target_vx),
                (self.vy, self.target_vy),
                (self.vz, self.target_vz),
            ]
        ):
            diff = tv - v
            max_dv = turn_rate * dt
            if abs(diff) <= max_dv:
                new_v = tv
            else:
                new_v = v + max_dv * (1.0 if diff > 0 else -1.0)
            if i == 0:
                self.vx = new_v
            elif i == 1:
                self.vy = new_v
            elif i == 2:
                self.vz = new_v

        # 3D Unpredictability calculation (rate of velocity vector direction change)
        old_speed = norm3(old_v)
        curr_speed = norm3((self.vx, self.vy, self.vz))
        if old_speed > 1e-3 and curr_speed > 1e-3:
            dot_prod = max(
                -1.0,
                min(
                    1.0,
                    (old_v[0] * self.vx + old_v[1] * self.vy + old_v[2] * self.vz)
                    / (old_speed * curr_speed),
                ),
            )
            angular_change = math.acos(dot_prod)
            self.unpredictability = angular_change / dt
        else:
            self.unpredictability = 0.0

        # Autonomous free random motion for EVERY target (independent, not clustered/flocked)
        self.evasive_timer -= dt
        if self.evasive_timer <= 0:
            # Pick a new independent random 3D direction vector
            az = random.uniform(-math.pi, math.pi)
            el = random.uniform(math.radians(-22.0), math.radians(22.0))
            s = self.speed
            self.target_vx = s * math.cos(el) * math.cos(az)
            self.target_vy = s * math.cos(el) * math.sin(az)
            self.target_vz = s * math.sin(el)
            self.evasive_timer = random.uniform(3.5, 7.5)

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Boundary bounce / reflection
        if self.x <= min_x:
            self.vx = abs(self.vx)
            self.target_vx = abs(self.target_vx)
            self.x = min_x
        elif self.x >= max_x:
            self.vx = -abs(self.vx)
            self.target_vx = -abs(self.target_vx)
            self.x = max_x

        if self.y <= min_y:
            self.vy = abs(self.vy)
            self.target_vy = abs(self.target_vy)
            self.y = min_y
        elif self.y >= max_y:
            self.vy = -abs(self.vy)
            self.target_vy = -abs(self.target_vy)
            self.y = max_y

        if self.z <= min_z:
            self.vz = abs(self.vz)
            self.target_vz = abs(self.target_vz)
            self.z = min_z
        elif self.z >= max_z:
            self.vz = -abs(self.vz)
            self.target_vz = -abs(self.target_vz)
            self.z = max_z

        # 3D Solid Obstacle Collision (Mountains / Terrain)
        # Prevents targets from passing through mountains by hard surface extrusion
        r = 12.0  # target collision radius
        for obs in obstacles:
            if (
                obs.min_x - r <= self.x <= obs.max_x + r
                and obs.min_y - r <= self.y <= obs.max_y + r
                and obs.min_z <= self.z <= obs.max_z + r
            ):
                # Determine minimum penetration distance to boundary faces
                d_top = abs((obs.max_z + r) - self.z)
                d_left = abs(self.x - (obs.min_x - r))
                d_right = abs((obs.max_x + r) - self.x)
                d_front = abs(self.y - (obs.min_y - r))
                d_back = abs((obs.max_y + r) - self.y)

                min_d = min(d_top, d_left, d_right, d_front, d_back)

                if min_d == d_top:
                    self.z = obs.max_z + r + 1.0
                    self.vz = abs(self.vz) * 0.8 + 4.0
                    self.target_vz = abs(self.target_vz)
                elif min_d == d_left:
                    self.x = obs.min_x - r - 1.0
                    self.vx = -abs(self.vx)
                    self.target_vx = -abs(self.target_vx)
                elif min_d == d_right:
                    self.x = obs.max_x + r + 1.0
                    self.vx = abs(self.vx)
                    self.target_vx = abs(self.target_vx)
                elif min_d == d_front:
                    self.y = obs.min_y - r - 1.0
                    self.vy = -abs(self.vy)
                    self.target_vy = -abs(self.target_vy)
                else:
                    self.y = obs.max_y + r + 1.0
                    self.vy = abs(self.vy)
                    self.target_vy = abs(self.target_vy)

        # Operational Uncertainty calculation & 1.1s Weapons Lock
        if not self.is_observed:
            self.time_since_observed += dt
            self.consecutive_obs_time = 0.0
            self.lock_acquired = False
            # Realistic uncertainty accumulation based on 3D velocity and maneuver rate
            speed_3d = norm3((self.vx, self.vy, self.vz))
            maneuver_growth = 1.0 + (self.unpredictability * 0.5)
            self.uncertainty += speed_3d * dt * 0.045 * maneuver_growth
        else:
            self.time_since_observed = 0.0
            self.consecutive_obs_time += dt
            # Active radar illumination collapses position uncertainty
            self.uncertainty = max(0.0, self.uncertainty - 40.0 * dt)
            # Weapons lock acquired if target is under radar view for 1.1 seconds
            if self.consecutive_obs_time >= 1.1 and self.uncertainty <= 12.0:
                self.lock_acquired = True

    def set_priority_class(self, new_class):
        """
        Dynamically changes target priority class with instant visual and RCS update.
        - Class 1: High Priority / Red (#ff003b) / Stealth (RCS: 0.15 m²)
        - Class 2: Medium Priority / Orange (#ff7700) / Armed Drone (RCS: 1.0 m²)
        - Class 3: Low Priority / Yellow (#ffee00) / Recon/Transport (RCS: 4.5 m²)
        """
        self.priority_class = int(new_class)
        if self.priority_class == 1:
            self.color = "#ff003b"
            self.rcs = 0.15
        elif self.priority_class == 2:
            self.color = "#ff7700"
            self.rcs = 1.0
        else:
            self.color = "#ffee00"
            self.rcs = 4.5
        self.cluster_color = self.color

    @property
    def track_state(self):
        """
        Operational track quality based on 3D spatial position uncertainty.
        - FIRM: High precision track (<8m), weapons lock eligible.
        - STABLE: Normal track-while-scan accuracy (8m-25m).
        - COAST: Extrapolated track, unobserved (25m-45m).
        - CRITICAL: Severe uncertainty (>45m), imminent track loss.
        """
        if self.uncertainty < 8.0:
            return "FIRM"
        elif self.uncertainty < 25.0:
            return "STABLE"
        elif self.uncertainty < 45.0:
            return "COAST"
        else:
            return "CRITICAL"


class TargetCluster:
    """
    Simulates a coordinated group of enemy targets (Drone Swarm, Fighter Strike Formation,
    or Cruise Missile Salvo) with Reynolds flocking and tactical split maneuvers.
    """

    def __init__(self, cluster_id, name, targets, color="#00e5ff"):
        self.cluster_id = cluster_id
        self.name = name
        self.targets = targets  # List of Target3D
        self.color = color

        # Tactical cluster behavior
        self.scatter_mode = False
        self.scatter_timer = 0.0
        self.mission_heading = random.uniform(-math.pi, math.pi)
        self.mission_pitch = random.uniform(math.radians(-5.0), math.radians(5.0))
        self.formation_spacing = 45.0  # desired spacing in meters

        for t in self.targets:
            t.cluster_id = self.cluster_id
            t.cluster_color = self.color

    @property
    def alive_targets(self):
        return [t for t in self.targets if not t.eliminated]

    def trigger_scatter(self):
        """Triggers high-G tactical dispersal / split maneuver."""
        self.scatter_mode = True
        self.scatter_timer = 6.0
        for i, t in enumerate(self.alive_targets):
            # Disperse radially outward in 3D
            angle = (2.0 * math.pi * i) / max(1, len(self.alive_targets))
            pitch = random.uniform(math.radians(-25.0), math.radians(25.0))
            burst_speed = t.speed * 1.35
            t.target_vx = burst_speed * math.cos(pitch) * math.cos(angle)
            t.target_vy = burst_speed * math.cos(pitch) * math.sin(angle)
            t.target_vz = burst_speed * math.sin(pitch)

    def get_threat_score(self):
        """Calculates tactical threat rating for this swarm based on count and class priority."""
        alive = self.alive_targets
        if not alive:
            return 0.0
        return sum((4 - t.priority_class) * 12.0 for t in alive)

    def scatter_on_member_destroyed(self, destroyed_id):
        """When a swarm drone is destroyed, surviving swarm members execute an evasive split."""
        self.scatter_mode = True
        self.scatter_timer = 5.0
        survivors = [t for t in self.alive_targets if t.id != destroyed_id]
        for i, t in enumerate(survivors):
            angle = (2.0 * math.pi * i) / max(1, len(survivors))
            pitch = random.uniform(math.radians(-25.0), math.radians(25.0))
            burst_speed = t.speed * 1.4
            t.target_vx = burst_speed * math.cos(pitch) * math.cos(angle)
            t.target_vy = burst_speed * math.cos(pitch) * math.sin(angle)
            t.target_vz = burst_speed * math.sin(pitch)
            t.evasive_timer = 0.0

    def get_centroid(self):
        alive = self.alive_targets
        if not alive:
            return (0, 0, 0)
        cx = sum(t.x for t in alive) / len(alive)
        cy = sum(t.y for t in alive) / len(alive)
        cz = sum(t.z for t in alive) / len(alive)
        return (cx, cy, cz)

    def get_dispersion_radius(self):
        alive = self.alive_targets
        if len(alive) <= 1:
            return 10.0
        c = self.get_centroid()
        return max(dist3(t.position, c) for t in alive)

    def update_flocking(self, dt, carrier_aircraft, obstacles=None):
        alive = self.alive_targets
        if len(alive) == 0:
            return

        if self.scatter_mode:
            self.scatter_timer -= dt
            if self.scatter_timer <= 0:
                self.scatter_mode = False
            return  # In scatter mode, targets continue on their breakout vectors

        centroid = self.get_centroid()

        # Mission navigation vector: cluster tends to advance towards carrier aircraft sector
        dx_carrier = carrier_aircraft.x - centroid[0]
        dy_carrier = carrier_aircraft.y - centroid[1]
        dz_carrier = carrier_aircraft.z - centroid[2]
        dist_c = math.sqrt(
            dx_carrier * dx_carrier + dy_carrier * dy_carrier + dz_carrier * dz_carrier
        )

        mission_dir = (
            dx_carrier / max(1.0, dist_c),
            dy_carrier / max(1.0, dist_c),
            dz_carrier / max(1.0, dist_c),
        )

        # Flocking behaviors: Separation, Alignment, Cohesion, Obstacle Avoidance
        for t in alive:
            sep_v = (0.0, 0.0, 0.0)
            align_v = (0.0, 0.0, 0.0)
            coh_v = (0.0, 0.0, 0.0)
            neighbors = 0

            for other in alive:
                if other.id == t.id:
                    continue
                d = dist3(t.position, other.position)
                if d < 120.0:  # neighbor radius
                    neighbors += 1
                    # Separation
                    if d < self.formation_spacing and d > 1e-3:
                        push = (t.x - other.x, t.y - other.y, t.z - other.z)
                        push_norm = normalize3(push)
                        weight = (self.formation_spacing - d) / self.formation_spacing
                        sep_v = add3(sep_v, mul3(push_norm, weight * 40.0))
                    # Alignment
                    align_v = add3(align_v, other.velocity)
                    # Cohesion
                    coh_v = add3(coh_v, (other.x - t.x, other.y - t.y, other.z - t.z))

            # Combine flocking forces
            desired_vel = (t.vx, t.vy, t.vz)

            # Base mission vector
            desired_vel = add3(desired_vel, mul3(mission_dir, 15.0 * dt))

            if neighbors > 0:
                # Separation is strongest to prevent collisions
                desired_vel = add3(desired_vel, mul3(sep_v, 2.0 * dt))
                # Alignment
                avg_align = mul3(align_v, 1.0 / neighbors)
                desired_vel = add3(
                    desired_vel, mul3(sub3(avg_align, (t.vx, t.vy, t.vz)), 0.6 * dt)
                )
                # Cohesion
                avg_coh = mul3(coh_v, 1.0 / neighbors)
                desired_vel = add3(desired_vel, mul3(avg_coh, 0.4 * dt))

            # Terrain Mountain Avoidance
            if obstacles:
                for obs in obstacles:
                    if (
                        obs.min_x - 70 <= t.x <= obs.max_x + 70
                        and obs.min_y - 70 <= t.y <= obs.max_y + 70
                        and t.z <= obs.max_z + 50
                    ):
                        cx_m = (obs.min_x + obs.max_x) / 2.0
                        cy_m = (obs.min_y + obs.max_y) / 2.0
                        push_x = t.x - cx_m
                        push_y = t.y - cy_m
                        p_len = math.hypot(push_x, push_y)
                        if p_len > 1e-3:
                            desired_vel = add3(
                                desired_vel,
                                (
                                    push_x / p_len * 50.0 * dt,
                                    push_y / p_len * 50.0 * dt,
                                    30.0 * dt,
                                ),
                            )

            # Re-normalize speed
            desired_dir = normalize3(desired_vel)
            t.target_vx = desired_dir[0] * t.speed
            t.target_vy = desired_dir[1] * t.speed
            t.target_vz = desired_dir[2] * t.speed
