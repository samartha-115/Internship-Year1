import math
import random

from carrier_aircraft import CarrierAircraft
from cluster_targets import Target3D, TargetCluster
from math_3d import (
    cartesian_to_spherical,
    dist3,
    normalize_angle,
)
from radar_physics_3d import Obstacle3D, RadarPhysics3D


class RadarEnvironment3D:
    """
    Complete 3D Real-Time Airspace Environment for Intelligent Radar-Target Scheduling.
    Integrates:
    - Maneuvering Carrier Aircraft (Ownship radar platform)
    - 3D Radar Physics, SNR, Detection Depth, and Volumetric Beam Cones
    - 3D Search Airspace Geometric Redundancy Engine
    - Enemy Cluster Targets (swarms/flocks) and Solo Stealth Penetrators
    - 3D Terrain Mountain Massifs
    """

    def __init__(self, width=2000.0, length=2000.0, height=800.0):
        self.bounds = (0.0, width, 0.0, length, 0.0, height)
        self.width = width
        self.length = length
        self.height = height
        self.current_time = 0.0

        # Carrier Aircraft (Airborne Radar Platform patrolling the expanded airspace)
        self.carrier = CarrierAircraft(
            x=1000.0, y=1000.0, z=420.0, speed=75.0, heading=math.radians(35.0)
        )
        self.carrier.orbit_center = (1000.0, 1000.0)
        self.carrier.orbit_radius = 520.0

        # 3D Radar Physics Engine scaled for 2000m+ airspace (1.5x slew speed)
        self.radar = RadarPhysics3D(
            beamwidth=math.radians(6.5),
            max_detection_depth=2600.0,
            slew_rate_az=math.radians(240.0),  # 1.5x speed: 240 deg/sec
            slew_rate_el=math.radians(180.0),  # 1.5x speed: 180 deg/sec
        )

        # 3D Mountain Massifs / Terrain Obstacles with distinctly different heights
        self.obstacles = [
            Obstacle3D(
                min_x=320,
                max_x=660,
                min_y=400,
                max_y=760,
                max_z=460,
                name="Mt. Apex (Peak - 460m)",
            ),
            Obstacle3D(
                min_x=1320,
                max_x=1660,
                min_y=1220,
                max_y=1580,
                max_z=340,
                name="Mt. Sierra (High - 340m)",
            ),
            Obstacle3D(
                min_x=1440,
                max_x=1840,
                min_y=400,
                max_y=720,
                max_z=240,
                name="Titan Ridge (Mid - 240m)",
            ),
            Obstacle3D(
                min_x=420,
                max_x=720,
                min_y=1350,
                max_y=1680,
                max_z=150,
                name="Iron Mesa (Low - 150m)",
            ),
        ]

        # 3D Ground Trees scattered across open terrain
        from radar_physics_3d import Tree3D

        self.trees = []
        random.seed(101)  # Scenic realistic placement
        for _ in range(70):
            tx = random.uniform(90, 1910)
            ty = random.uniform(90, 1910)
            # Ensure tree is outside mountain footprints
            in_mountain = False
            for obs in self.obstacles:
                if (
                    obs.min_x - 30 <= tx <= obs.max_x + 30
                    and obs.min_y - 30 <= ty <= obs.max_y + 30
                ):
                    in_mountain = True
                    break
            if not in_mountain:
                h = random.uniform(50.0, 88.0)
                r = random.uniform(22.5, 37.5)
                col = random.choice(["#059669", "#10b981", "#047857", "#065f46"])
                self.trees.append(Tree3D(tx, ty, height=h, radius=r, color=col))
        random.seed()

        # Target Management: All targets move independently (no flocking)
        self.targets = []
        self.clusters = {}
        self.target_id_counter = 1
        self.beam_locked_on_id = None
        self.actively_targeted_id = None  # Target radar is actively pointing at
        self.last_blast_event = None      # Swarm salvo area-of-effect event

        # Explosions / Particle bursts for visual feedback
        self.explosions = []

        # Performance & Tracking Metrics
        self.metrics = {
            "eliminated": 0,
            "avg_uncertainty": 0.0,
            "redundancy_efficiency": 0.0,
            "active_in_for": 0,
            "locked_targets": 0,
        }

        # Initialize Default Battlefield Targets
        self._init_battlefield_scenario()

    def _init_battlefield_scenario(self):
        """
        Spawns 15 targets organized into tactical swarms and independent penetrators
        throughout the expanded 2000x2000x800 airspace.
        - Swarm Alpha (Strike): 4 High Priority Class 1 Targets (Red #ff003b)
        - Swarm Beta (Fighters): 4 Medium Priority Class 2 Targets (Orange #ff7700)
        - Swarm Gamma (Drones): 4 Low Priority Class 3 Targets (Yellow #ffee00)
        - 3 Independent Solo Stealth Penetrators (Class 1, 2, 3)
        """
        # Swarm Alpha: 4 High Priority Strike Fighters (Red - 1.5x speed)
        alpha_targets = []
        for i in range(4):
            x = 650.0 + random.uniform(-180, 180)
            y = 1450.0 + random.uniform(-180, 180)
            z = random.uniform(320, 680)
            s = random.uniform(82.5, 112.5)  # 1.5x speed: 82.5 - 112.5 m/s
            ang = random.uniform(-math.pi, math.pi)
            el = random.uniform(math.radians(-12), math.radians(12))
            vx = s * math.cos(el) * math.cos(ang)
            vy = s * math.cos(el) * math.sin(ang)
            vz = s * math.sin(el)
            t = Target3D(
                self.target_id_counter,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                priority_class=1,
                cluster_id="Alpha",
            )
            self.targets.append(t)
            alpha_targets.append(t)
            self.target_id_counter += 1
        self.clusters["Alpha"] = TargetCluster(
            "Alpha", "Swarm Alpha (Strike)", alpha_targets, color="#ff003b"
        )

        # Swarm Beta: 4 Medium Priority Escort Interceptors (Orange - 1.5x speed)
        beta_targets = []
        for i in range(4):
            x = 1450.0 + random.uniform(-200, 200)
            y = 700.0 + random.uniform(-200, 200)
            z = random.uniform(220, 560)
            s = random.uniform(67.5, 97.5)  # 1.5x speed: 67.5 - 97.5 m/s
            ang = random.uniform(-math.pi, math.pi)
            el = random.uniform(math.radians(-10), math.radians(10))
            vx = s * math.cos(el) * math.cos(ang)
            vy = s * math.cos(el) * math.sin(ang)
            vz = s * math.sin(el)
            t = Target3D(
                self.target_id_counter,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                priority_class=2,
                cluster_id="Beta",
            )
            self.targets.append(t)
            beta_targets.append(t)
            self.target_id_counter += 1
        self.clusters["Beta"] = TargetCluster(
            "Beta", "Swarm Beta (Fighters)", beta_targets, color="#ff7700"
        )

        # Swarm Gamma: 4 Low Priority Recon Drones (Yellow - 1.5x speed)
        gamma_targets = []
        for i in range(4):
            x = 1200.0 + random.uniform(-220, 220)
            y = 1500.0 + random.uniform(-220, 220)
            z = random.uniform(140, 420)
            s = random.uniform(52.5, 82.5)  # 1.5x speed: 52.5 - 82.5 m/s
            ang = random.uniform(-math.pi, math.pi)
            el = random.uniform(math.radians(-8), math.radians(8))
            vx = s * math.cos(el) * math.cos(ang)
            vy = s * math.cos(el) * math.sin(ang)
            vz = s * math.sin(el)
            t = Target3D(
                self.target_id_counter,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                priority_class=3,
                cluster_id="Gamma",
            )
            self.targets.append(t)
            gamma_targets.append(t)
            self.target_id_counter += 1
        self.clusters["Gamma"] = TargetCluster(
            "Gamma", "Swarm Gamma (Drones)", gamma_targets, color="#ffee00"
        )

        # 3 Independent Solo Penetrators (Class 1, 2, 3 - 1.5x speed)
        for p_cls in (1, 2, 3):
            x = random.uniform(250, 1750)
            y = random.uniform(250, 1750)
            z = random.uniform(200, 600)
            s = random.uniform(60.0, 105.0)  # 1.5x speed: 60 - 105 m/s
            ang = random.uniform(-math.pi, math.pi)
            el = random.uniform(math.radians(-12), math.radians(12))
            vx = s * math.cos(el) * math.cos(ang)
            vy = s * math.cos(el) * math.sin(ang)
            vz = s * math.sin(el)
            t = Target3D(
                self.target_id_counter,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                priority_class=p_cls,
                cluster_id=None,
            )
            self.targets.append(t)
            self.target_id_counter += 1

    def spawn_random_target(self, priority_class=None, cluster_id=None):
        """Spawns a new independent random target in the expanded airspace."""
        if priority_class is None:
            priority_class = random.choices([1, 2, 3], weights=[0.35, 0.40, 0.25])[0]

        spawn_x = random.choice([150.0, self.width - 150.0])
        spawn_y = random.uniform(150.0, self.length - 150.0)
        spawn_z = random.uniform(120.0, 680.0)

        s = random.uniform(60.0, 105.0)  # 1.5x speed: 60 - 105 m/s
        ang = random.uniform(-math.pi, math.pi)
        el = random.uniform(math.radians(-15), math.radians(15))
        vx = s * math.cos(el) * math.cos(ang)
        vy = s * math.cos(el) * math.sin(ang)
        vz = s * math.sin(el)

        t = Target3D(
            self.target_id_counter,
            spawn_x,
            spawn_y,
            spawn_z,
            vx,
            vy,
            vz,
            priority_class=priority_class,
            cluster_id=cluster_id,
        )
        self.targets.append(t)
        self.target_id_counter += 1
        if cluster_id and cluster_id in self.clusters:
            self.clusters[cluster_id].targets.append(t)
        return t

    def spawn_cluster(self, cluster_id=None):
        """Spawns 3 random targets across priority classes (Red, Orange, Yellow)."""
        t1 = self.spawn_random_target(priority_class=1, cluster_id=cluster_id)
        t2 = self.spawn_random_target(priority_class=2, cluster_id=cluster_id)
        t3 = self.spawn_random_target(priority_class=3, cluster_id=cluster_id)
        return [t1, t2, t3]

    def trigger_cluster_scatter(self, cluster_id=None):
        """Triggers dynamic 3D evasive maneuvers for targets or a specific swarm."""
        if cluster_id and cluster_id in self.clusters:
            self.clusters[cluster_id].trigger_scatter()
        else:
            for t in self.targets:
                t.vx += random.uniform(-25, 25)
                t.vy += random.uniform(-25, 25)
                t.vz += random.uniform(-15, 15)
                t.evasive_timer = 0.0

    def set_target_class(self, target_id, new_class):
        """Manually updates the priority class of a target with instant visual/RCS change."""
        for t in self.targets:
            if t.id == target_id:
                t.set_priority_class(new_class)
                return True
        return False

    def target_swarm(self, cluster_id):
        """Returns the ID of the highest threat target in the specified cluster."""
        if cluster_id in self.clusters:
            alive = self.clusters[cluster_id].alive_targets
            if alive:
                return min(
                    alive, key=lambda t: (t.priority_class, t.time_since_observed)
                ).id
        return None

    def fire_swarm_salvo(self, target_id=None, blast_radius=450.0):
        """
        Executes a synchronized swarm salvo / cluster area-of-effect intercept.
        Eliminates the locked main target AND all active enemy targets located within
        blast_radius (default 450m) of the locked main target.
        Returns list of eliminated target IDs.
        """
        main_t = None
        if target_id is not None:
            for t in self.targets:
                if t.id == target_id and not t.eliminated:
                    main_t = t
                    break

        if main_t is None:
            # Fallback to any locked target in airspace
            locked_candidates = [
                t for t in self.targets if t.lock_acquired and not t.eliminated
            ]
            if locked_candidates:
                main_t = locked_candidates[0]

        if main_t is None or not main_t.lock_acquired:
            return []

        main_pos = main_t.position

        # Swarm Fire AOE: Find all targets within blast radius of the locked main target
        candidates = [
            t
            for t in self.targets
            if not t.eliminated and dist3(main_pos, t.position) <= blast_radius
        ]
        if main_t not in candidates:
            candidates.insert(0, main_t)

        destroyed = []
        for t in candidates:
            if self.destroy_target(t.id):
                destroyed.append(t.id)

        self.last_blast_event = {
            "center": main_pos,
            "radius": blast_radius,
            "time": self.current_time,
            "count": len(destroyed),
        }

        return destroyed

    def is_line_of_sight_clear(self, target):
        """Checks 3D ray from Carrier Aircraft to Target against 3D Mountain Massifs."""
        p_carrier = self.carrier.position
        p_target = target.position
        for obs in self.obstacles:
            if obs.blocks_line_of_sight(p_carrier, p_target):
                return False
        return True

    def destroy_target(self, target_id):
        """Destroys target and spawns 3D particle explosion, triggering reactive swarm scatter."""
        for t in self.targets:
            if t.id == target_id and not t.eliminated:
                t.eliminated = True
                self.metrics["eliminated"] += 1
                c_id = t.cluster_id

                # Trigger tactical scatter for remaining swarm members
                if c_id and c_id in self.clusters:
                    self.clusters[c_id].scatter_on_member_destroyed(t.id)

                # Add 3D explosion animation
                self.explosions.append(
                    {
                        "pos": t.position,
                        "radius": 14.0,
                        "max_radius": 65.0,
                        "life": 1.0,
                        "color": t.color,
                    }
                )
                # Spawn replacement target at perimeter so airspace remains populated
                self.spawn_random_target(
                    priority_class=t.priority_class, cluster_id=c_id
                )
                return True
        return False

    def step(self, dt, target_id_to_track):
        self.current_time += dt
        self.beam_locked_on_id = None

        # 1. Update Carrier Aircraft Kinematics (Point 3) with solid obstacle avoidance
        self.carrier.update(dt, self.bounds, self.obstacles)
        carrier_pos = self.carrier.position

        # 2. Update Kinematics & Collisions for all Targets freely and independently
        for t in self.targets:
            t.update_kinematics(dt, self.bounds, self.obstacles)

        # 4. Target to track determination
        target_to_track = None
        for t in self.targets:
            if t.id == target_id_to_track and not t.eliminated:
                target_to_track = t
                break

        # Radar must ALWAYS point at/target an active target at any given point of time
        if target_to_track is None:
            alive_targets = [t for t in self.targets if not t.eliminated]
            if alive_targets:
                in_for = [
                    t
                    for t in alive_targets
                    if self.carrier.is_in_field_of_regard(t.position)[0]
                ]
                if in_for:
                    target_to_track = min(
                        in_for,
                        key=lambda t: (
                            t.priority_class,
                            dist3(carrier_pos, t.position),
                        ),
                    )
                else:
                    target_to_track = min(
                        alive_targets,
                        key=lambda t: dist3(carrier_pos, t.position),
                    )

        self.actively_targeted_id = target_to_track.id if target_to_track else None

        # Compute required 3D beam angles in world coordinates
        target_range = 0.0
        if target_to_track:
            dx = target_to_track.x - carrier_pos[0]
            dy = target_to_track.y - carrier_pos[1]
            dz = target_to_track.z - carrier_pos[2]
            target_range, desired_beam_az, desired_beam_el = cartesian_to_spherical(
                dx, dy, dz
            )
        else:
            # Only if all targets are eliminated
            desired_beam_az = self.carrier.heading
            desired_beam_el = 0.0

        # 5. Slew 3D Radar Beam (Point 2)
        beam_aligned = self.radar.slew_towards(desired_beam_az, desired_beam_el, dt)

        # 6. Evaluate Target Detection & Dwell (Point 2)
        illuminated_ids = set()

        if target_to_track:
            # Check A: Carrier Field of Regard (FOR)
            in_for, body_az, body_el, _r_body = self.carrier.is_in_field_of_regard(
                target_to_track.position
            )

            # Check B: 3D Line of Sight (LOS)
            los_clear = (
                self.is_line_of_sight_clear(target_to_track) if in_for else False
            )

            # Check C: Detection Depth and Radar Range Equation (SNR with spatial uncertainty)
            off_az = abs(normalize_angle(desired_beam_az - self.radar.beam_az))
            off_el = abs(desired_beam_el - self.radar.beam_el)
            off_boresight = math.hypot(off_az, off_el)

            snr_db = self.radar.calculate_snr(
                target_range,
                target_to_track.rcs,
                off_boresight,
                uncertainty_m=target_to_track.uncertainty,
            )
            detection_depth_max = self.radar.calculate_detection_depth(
                target_to_track.rcs
            )
            self.radar.current_detection_depth = detection_depth_max
            p_d = self.radar.calculate_detection_probability(snr_db)

            # Record geometric redundancy dwell in 3D airspace
            self.radar.redundancy_mgr.record_beam_dwell(
                body_az, body_el, self.radar.beamwidth, self.current_time
            )

            # Probabilistic Detection Roll based on SNR and Detection Depth
            if (
                beam_aligned
                and in_for
                and los_clear
                and (target_range <= detection_depth_max)
            ):
                # Roll against detection probability Pd
                if random.random() <= p_d:
                    target_to_track.is_observed = True
                    self.beam_locked_on_id = target_to_track.id
                    illuminated_ids.add(target_to_track.id)
                else:
                    target_to_track.is_observed = False
            else:
                target_to_track.is_observed = False

        # Multi-target volumetric dwell: all targets caught inside the 3D beam cone
        # receive beam illumination and uncertainty collapse
        if beam_aligned:
            for t in self.targets:
                if t.eliminated or t.id == self.beam_locked_on_id:
                    continue
                dx = t.x - carrier_pos[0]
                dy = t.y - carrier_pos[1]
                dz = t.z - carrier_pos[2]
                r_t, az_t, el_t = cartesian_to_spherical(dx, dy, dz)

                in_for_t, _, _, _ = self.carrier.is_in_field_of_regard(t.position)
                if not in_for_t or not self.is_line_of_sight_clear(t):
                    continue

                off_az_t = abs(normalize_angle(az_t - self.radar.beam_az))
                off_el_t = abs(el_t - self.radar.beam_el)
                off_b_t = math.hypot(off_az_t, off_el_t)

                if off_b_t <= self.radar.beamwidth * 0.95:
                    d_max = self.radar.calculate_detection_depth(t.rcs)
                    if r_t <= d_max:
                        snr_t = self.radar.calculate_snr(
                            r_t, t.rcs, off_b_t, uncertainty_m=t.uncertainty
                        )
                        pd_t = self.radar.calculate_detection_probability(snr_t)
                        if random.random() <= pd_t:
                            t.is_observed = True
                            illuminated_ids.add(t.id)

        # Reset is_observed for all non-illuminated targets
        for t in self.targets:
            if t.id not in illuminated_ids:
                t.is_observed = False

        # 7. Update Particle Explosions
        for exp in self.explosions:
            exp["life"] -= dt * 1.8
            exp["radius"] += dt * 38.0
        self.explosions = [e for e in self.explosions if e["life"] > 0.0]

        # 8. Compute System Aggregate Metrics
        alive_targets = [t for t in self.targets if not t.eliminated]
        if alive_targets:
            self.metrics["avg_uncertainty"] = sum(
                t.uncertainty for t in alive_targets
            ) / len(alive_targets)
            in_for_count = 0
            locked_count = 0
            for t in alive_targets:
                in_for, _, _, _ = self.carrier.is_in_field_of_regard(t.position)
                if in_for:
                    in_for_count += 1
                if t.lock_acquired:
                    locked_count += 1
            self.metrics["active_in_for"] = in_for_count
            self.metrics["locked_targets"] = locked_count
        else:
            self.metrics["avg_uncertainty"] = 0.0
            self.metrics["active_in_for"] = 0
            self.metrics["locked_targets"] = 0

        self.metrics["redundancy_efficiency"] = (
            self.radar.redundancy_mgr.get_overall_redundancy_efficiency()
        )

        # Clean up eliminated targets
        self.targets = [t for t in self.targets if not t.eliminated]
