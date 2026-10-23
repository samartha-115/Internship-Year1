import math

from math_3d import (
    cartesian_to_spherical,
    normalize_angle,
    rotate_world_to_body,
)


class CarrierAircraft:
    """
    Airborne Radar Platform (Ownship).
    Simulates a maneuvering carrier/fighter aircraft equipped with an advanced
    airborne fire-control/surveillance radar.
    """

    def __init__(self, x=1000.0, y=1000.0, z=420.0, speed=75.0, heading=0.0):
        # 3D Position
        self.x = x
        self.y = y
        self.z = z

        # Kinematics
        self.speed = speed
        self.target_speed = speed
        self.heading = heading  # Yaw (psi) in radians
        self.target_heading = heading
        self.pitch = 0.0  # Theta in radians
        self.target_pitch = 0.0
        self.roll = 0.0  # Bank angle (gamma) in radians
        self.target_roll = 0.0

        # Flight limits
        self.max_roll = math.radians(45.0)  # 45 deg max bank
        self.roll_rate = math.radians(60.0)  # 60 deg/sec roll speed
        self.max_pitch = math.radians(25.0)  # 25 deg climb/dive
        self.pitch_rate = math.radians(20.0)

        # Maneuver Modes
        self.mode = "ORBIT_CAP"  # ORBIT_CAP, EVASIVE_S_TURNS, STRAIGHT_LEVEL, MANUAL
        self.maneuver_timer = 0.0
        self.orbit_radius = 520.0
        self.orbit_center = (1000.0, 1000.0)

        # Radar Field of Regard (FOR) limits relative to aircraft body frame
        # Azimuth: +/- 75 degrees, Elevation: -25 to +55 degrees
        self.for_azimuth_limit = math.radians(75.0)
        self.for_elevation_min = math.radians(-25.0)
        self.for_elevation_max = math.radians(55.0)

        # History for 3D flight path trail
        self.history = []

    @property
    def position(self):
        return (self.x, self.y, self.z)

    @property
    def velocity(self):
        # Velocity in world coordinates
        horiz = self.speed * math.cos(self.pitch)
        vx = horiz * math.cos(self.heading)
        vy = horiz * math.sin(self.heading)
        vz = self.speed * math.sin(self.pitch)
        return (vx, vy, vz)

    def set_mode(self, mode):
        self.mode = mode
        self.maneuver_timer = 0.0

    def update(self, dt, airspace_bounds, obstacles=None):
        min_x, max_x, min_y, max_y, min_z, max_z = airspace_bounds

        self.maneuver_timer += dt

        # Autonomous flight logic based on mode
        if self.mode == "ORBIT_CAP":
            # Orbit around center in a circular/elliptical racetrack
            dx = self.x - self.orbit_center[0]
            dy = self.y - self.orbit_center[1]
            current_dist = math.hypot(dx, dy)
            tangent_angle = math.atan2(dy, dx) + math.pi / 2.0

            # Steer towards tangent with radial correction
            rad_corr = -0.3 * (current_dist - self.orbit_radius) / self.orbit_radius
            desired_h = normalize_angle(tangent_angle + rad_corr)
            self.target_heading = desired_h
            self.target_pitch = 0.0

        elif self.mode == "EVASIVE_S_TURNS":
            # Periodic tactical weaving (period ~ 6 seconds)
            turn_phase = math.sin(self.maneuver_timer * (2 * math.pi / 6.0))
            self.target_roll = turn_phase * math.radians(35.0)
            self.target_pitch = math.sin(self.maneuver_timer * 0.5) * math.radians(8.0)

        elif self.mode == "STRAIGHT_LEVEL":
            self.target_roll = 0.0
            self.target_pitch = 0.0

        elif self.mode == "MANUAL":
            # Handled directly by UI sliders
            pass

        # 3D Obstacle Terrain Avoidance System (TAWS) for Carrier Pilot
        if obstacles:
            for obs in obstacles:
                buf = 85.0
                if (
                    obs.min_x - buf <= self.x <= obs.max_x + buf
                    and obs.min_y - buf <= self.y <= obs.max_y + buf
                    and self.z <= obs.max_z + 60.0
                ):
                    # Proximity alert: climb over the peak and steer away
                    self.target_pitch = max(self.target_pitch, math.radians(20.0))
                    cx_m = (obs.min_x + obs.max_x) / 2.0
                    cy_m = (obs.min_y + obs.max_y) / 2.0
                    away_angle = math.atan2(self.y - cy_m, self.x - cx_m)
                    self.target_heading = away_angle

        # Coordinated turn physics:
        # If target_heading is set, compute required bank angle
        if self.mode in ("ORBIT_CAP", "STRAIGHT_LEVEL"):
            h_diff = normalize_angle(self.target_heading - self.heading)
            # Desired bank angle proportional to heading error
            desired_bank = max(-self.max_roll, min(self.max_roll, h_diff * 1.5))
            self.target_roll = desired_bank

        # Smooth roll response
        roll_diff = self.target_roll - self.roll
        max_d_roll = self.roll_rate * dt
        if abs(roll_diff) <= max_d_roll:
            self.roll = self.target_roll
        else:
            self.roll += max_d_roll * (1.0 if roll_diff > 0 else -1.0)

        # Smooth pitch response
        pitch_diff = self.target_pitch - self.pitch
        max_d_pitch = self.pitch_rate * dt
        if abs(pitch_diff) <= max_d_pitch:
            self.pitch = self.target_pitch
        else:
            self.pitch += max_d_pitch * (1.0 if pitch_diff > 0 else -1.0)

        # Turn rate from bank angle: g * tan(roll) / V
        g = 9.81
        if abs(self.speed) > 1.0:
            turn_rate = (g * math.tan(self.roll)) / self.speed
        else:
            turn_rate = 0.0
        self.heading = normalize_angle(self.heading + turn_rate * dt)

        # Kinematic update
        vx, vy, vz = self.velocity
        self.x += vx * dt
        self.y += vy * dt
        self.z += vz * dt

        # Hard 3D Physical Barrier: Pilot CANNOT pass through mountains
        if obstacles:
            r_c = 20.0  # aircraft collision envelope
            for obs in obstacles:
                if (
                    obs.min_x - r_c <= self.x <= obs.max_x + r_c
                    and obs.min_y - r_c <= self.y <= obs.max_y + r_c
                    and self.z <= obs.max_z + r_c
                ):
                    # Extrude out immediately
                    d_top = abs((obs.max_z + r_c) - self.z)
                    d_left = abs(self.x - (obs.min_x - r_c))
                    d_right = abs((obs.max_x + r_c) - self.x)
                    d_front = abs(self.y - (obs.min_y - r_c))
                    d_back = abs((obs.max_y + r_c) - self.y)

                    min_d = min(d_top, d_left, d_right, d_front, d_back)

                    if min_d == d_top or self.z >= obs.max_z - 30.0:
                        self.z = obs.max_z + r_c + 2.0
                        self.pitch = max(math.radians(6.0), self.pitch)
                    elif min_d == d_left:
                        self.x = obs.min_x - r_c - 2.0
                    elif min_d == d_right:
                        self.x = obs.max_x + r_c + 2.0
                    elif min_d == d_front:
                        self.y = obs.min_y - r_c - 2.0
                    else:
                        self.y = obs.max_y + r_c + 2.0

        # Airspace boundary avoidance
        margin = 100.0
        if (
            self.x < min_x + margin
            and math.cos(self.heading) < 0
            or self.x > max_x - margin
            and math.cos(self.heading) > 0
        ):
            self.heading = normalize_angle(math.pi - self.heading)

        if (
            self.y < min_y + margin
            and math.sin(self.heading) < 0
            or self.y > max_y - margin
            and math.sin(self.heading) > 0
        ):
            self.heading = normalize_angle(-self.heading)

        self.z = max(min_z + 40.0, min(max_z - 40.0, self.z))

        # Record trail
        self.history.append((self.x, self.y, self.z))
        if len(self.history) > 35:
            self.history.pop(0)

    def get_target_body_angles(self, target_world_pos):
        """
        Calculates Azimuth and Elevation of a target in the carrier aircraft's
        BODY frame (relative to aircraft nose).
        """
        # Vector from carrier to target in world coordinates
        rel_world = (
            target_world_pos[0] - self.x,
            target_world_pos[1] - self.y,
            target_world_pos[2] - self.z,
        )
        # Rotate into body frame
        rel_body = rotate_world_to_body(rel_world, self.heading, self.pitch, self.roll)
        r, az, el = cartesian_to_spherical(rel_body[0], rel_body[1], rel_body[2])
        return r, az, el, rel_body

    def is_in_field_of_regard(self, target_world_pos):
        """
        Checks if the target is within the radar antenna's mechanical/electronic
        gimbal scan limits (Field of Regard) relative to the aircraft nose.
        """
        r, az, el, rel_body = self.get_target_body_angles(target_world_pos)
        # Target must be in forward hemisphere and within azimuth / elevation limits
        if rel_body[0] <= 0:  # Behind the radar radome
            return False, az, el, r
        if abs(az) > self.for_azimuth_limit:
            return False, az, el, r
        if el < self.for_elevation_min or el > self.for_elevation_max:
            return False, az, el, r
        return True, az, el, r
