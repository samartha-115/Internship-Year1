import math

from math_3d import (
    cross3,
    normalize3,
    normalize_angle,
    ray_intersects_aabb,
)


class Obstacle3D:
    """
    3D Mountain / Terrain massif obstacle that blocks radar line of sight.
    Defined by a 3D bounding box [min_x, max_x] x [min_y, max_y] x [min_z, max_z].
    """

    def __init__(self, min_x, max_x, min_y, max_y, max_z, name="Mountain"):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = 0.0
        self.max_z = max_z
        self.name = name
        self._solid_faces = self._build_solid_faces()

    @property
    def box_min(self):
        return (self.min_x, self.min_y, self.min_z)

    @property
    def box_max(self):
        return (self.max_x, self.max_y, self.max_z)

    def blocks_line_of_sight(self, p1, p2):
        """Returns True if the 3D ray segment from p1 to p2 penetrates this obstacle."""
        return ray_intersects_aabb(p1, p2, self.box_min, self.box_max)

    def get_wireframe_vertices(self):
        """Returns vertices of the 3D mountain block for 3D rendering."""
        x0, x1 = self.min_x, self.max_x
        y0, y1 = self.min_y, self.max_y
        z0, z1 = self.min_z, self.max_z

        # 8 box corners
        return [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),  # Base (0..3)
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),  # Top (4..7)
        ]

    def _build_solid_faces(self):
        """Precomputes the 3D polygon faces of the mountain for solid 3D rendering."""
        x0, x1 = self.min_x, self.max_x
        y0, y1 = self.min_y, self.max_y
        z0, z1 = self.min_z, self.max_z

        v = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),  # Base (0..3)
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),  # Top (4..7)
        ]

        return [
            # Top face (illuminated stone plateau)
            ([v[4], v[5], v[6], v[7]], (0, 0, 1), "#334155", "#64748b"),
            # South face
            ([v[0], v[1], v[5], v[4]], (0, -1, 0), "#1e293b", "#475569"),
            # East face
            ([v[1], v[2], v[6], v[5]], (1, 0, 0), "#182234", "#334155"),
            # North face
            ([v[2], v[3], v[7], v[6]], (0, 1, 0), "#151d2c", "#2d3748"),
            # West face
            ([v[3], v[0], v[4], v[7]], (-1, 0, 0), "#233246", "#475569"),
        ]

    def get_solid_faces(self):
        """Returns the precomputed 3D polygon faces of the mountain."""
        return self._solid_faces


class Tree3D:
    """
    3D Ground Tree with trunk and 3D tiered canopy foliage (2.5x enlarged realistic scale).
    """

    def __init__(self, x, y, height=65.0, radius=27.5, color="#10b981"):
        self.x = x
        self.y = y
        self.height = height
        self.radius = radius
        self.color = color

    def get_geometry(self):
        """
        Returns (base_point, canopy_base_points, apex_point).
        """
        base = (self.x, self.y, 0.0)
        trunk_top = (self.x, self.y, self.height * 0.35)
        apex = (self.x, self.y, self.height)
        r = self.radius
        h_base = self.height * 0.35
        canopy = [
            (self.x - r, self.y - r, h_base),
            (self.x + r, self.y - r, h_base),
            (self.x + r, self.y + r, h_base),
            (self.x - r, self.y + r, h_base),
        ]
        return base, trunk_top, canopy, apex


class SearchAirspaceRedundancy:
    """
    Models 3D Search Airspace Geometric Redundancy.
    Divides the 3D angular Field of Regard (Azimuth x Elevation) into a lattice of
    volumetric search cells and computes beam overlap factors and coverage redundancy.
    """

    DEFAULT_AZ_RANGE = (-math.radians(70), math.radians(70))
    DEFAULT_EL_RANGE = (-math.radians(20), math.radians(45))

    def __init__(self, az_range=None, el_range=None, az_divisions=12, el_divisions=6):
        if az_range is None:
            az_range = self.DEFAULT_AZ_RANGE
        if el_range is None:
            el_range = self.DEFAULT_EL_RANGE
        self.min_az, self.max_az = az_range
        self.min_el, self.max_el = el_range
        self.az_divs = az_divs = az_divisions
        self.el_divs = el_divs = el_divisions

        # 2D grid of cells (azimuth x elevation)
        self.cell_w_az = (self.max_az - self.min_az) / az_divs
        self.cell_h_el = (self.max_el - self.min_el) / el_divs

        # Redundancy matrix: counts how many overlapping beam dwells cover each cell
        self.coverage_counts = [[0 for _ in range(el_divs)] for _ in range(az_divs)]
        self.last_scanned = [[-999.0 for _ in range(el_divs)] for _ in range(az_divs)]

        # Nominal beam overlap factor (typically 0.707 for 3dB crossover in 3D radar)
        self.overlap_factor = 0.72

    def record_beam_dwell(self, beam_az, beam_el, beamwidth, sim_time):
        """
        When the radar dwells at (beam_az, beam_el), updates the geometric
        redundancy distribution across all affected angular search cells.
        """
        bw_rad = beamwidth
        for i in range(self.az_divs):
            c_az = self.min_az + (i + 0.5) * self.cell_w_az
            d_az = abs(normalize_angle(c_az - beam_az))
            if d_az > bw_rad * 1.5:
                continue
            for j in range(self.el_divs):
                c_el = self.min_el + (j + 0.5) * self.cell_h_el
                d_el = abs(c_el - beam_el)
                if d_el > bw_rad * 1.5:
                    continue

                # Angular distance from beam center normalized by beamwidth
                ang_dist = math.hypot(d_az, d_el)
                if ang_dist <= bw_rad:
                    self.coverage_counts[i][j] += 1
                    self.last_scanned[i][j] = sim_time

    def get_cell_redundancy(self, az, el, sim_time):
        """
        Returns the geometric redundancy index (0.0 to 1.0+) for a target at (az, el).
        Higher redundancy means this sector is heavily covered by overlapping beams.
        """
        if az < self.min_az or az > self.max_az or el < self.min_el or el > self.max_el:
            return 0.0

        i = int((az - self.min_az) / self.cell_w_az)
        j = int((el - self.min_el) / self.cell_h_el)
        i = max(0, min(self.az_divs - 1, i))
        j = max(0, min(self.el_divs - 1, j))

        cov = self.coverage_counts[i][j]
        time_since = sim_time - self.last_scanned[i][j]

        # Decay redundancy over time if sector was not refreshed
        freshness = math.exp(-time_since / 6.0) if time_since >= 0 else 0.0
        redundancy_index = (cov * self.overlap_factor) * freshness
        return redundancy_index

    def get_overall_redundancy_efficiency(self):
        """
        Computes the ratio of useful coverage to over-sampled redundant dwells.
        Higher efficiency means the radar is distributing search beams evenly.
        """
        total_cells = self.az_divs * self.el_divs
        covered = sum(1 for row in self.coverage_counts for val in row if val > 0)
        total_dwells = sum(val for row in self.coverage_counts for val in row)
        if total_dwells == 0:
            return 0.0
        coverage_ratio = covered / total_cells
        # Efficiency penalizes clustering all dwells into the same cells
        return min(1.0, coverage_ratio * (covered / max(1, total_dwells * 0.4)))


class RadarPhysics3D:
    """
    Realistic 3D Radar Physics Engine.
    Implements:
    - 3D Beam Slew (Azimuth & Elevation mechanical / electronic steering)
    - Radar Range Equation (SNR vs Range^4, RCS, atmospheric attenuation)
    - Detection Depth [R_min, R_max(RCS, SNR)]
    - Volumetric 3D Beam Cone / Footprint
    - Detection Probability P_d
    """

    DEFAULT_BEAMWIDTH = math.radians(6.5)
    DEFAULT_SLEW_AZ = math.radians(240.0)  # 1.5x speed: 240 deg/sec
    DEFAULT_SLEW_EL = math.radians(180.0)  # 1.5x speed: 180 deg/sec

    def __init__(
        self,
        beamwidth=None,
        max_detection_depth=2600.0,
        min_range=50.0,
        slew_rate_az=None,
        slew_rate_el=None,
    ):
        if beamwidth is None:
            beamwidth = self.DEFAULT_BEAMWIDTH
        if slew_rate_az is None:
            slew_rate_az = self.DEFAULT_SLEW_AZ
        if slew_rate_el is None:
            slew_rate_el = self.DEFAULT_SLEW_EL
        self.beamwidth = beamwidth
        self.max_nominal_depth = max_detection_depth
        self.min_range = min_range
        self.slew_rate_az = slew_rate_az
        self.slew_rate_el = slew_rate_el

        # Current 3D Beam pointing direction in World Coordinates (Azimuth, Elevation)
        self.beam_az = 0.0
        self.beam_el = 0.0
        self.current_detection_depth = max_detection_depth

        # Radar Equation Parameters scaled for 2000m+ tactical airspace
        # Reference: SNR = 18 dB for RCS = 1.0 m^2 at R = 1400m
        self.snr_ref_db = 18.0
        self.r_ref = 1400.0
        self.rcs_ref = 1.0
        self.snr_threshold_db = 10.0  # Min SNR for 50% detection probability

        # Redundancy manager
        self.redundancy_mgr = SearchAirspaceRedundancy()

    def slew_towards(self, target_az, target_el, dt):
        """
        Slews the 3D radar beam in azimuth and elevation simultaneously towards
        the commanded angles. Returns True if beam has aligned within beamwidth.
        """
        d_az = normalize_angle(target_az - self.beam_az)
        d_el = target_el - self.beam_el

        max_d_az = self.slew_rate_az * dt
        max_d_el = self.slew_rate_el * dt

        if abs(d_az) <= max_d_az:
            self.beam_az = target_az
        else:
            self.beam_az = normalize_angle(
                self.beam_az + max_d_az * (1.0 if d_az > 0 else -1.0)
            )

        if abs(d_el) <= max_d_el:
            self.beam_el = target_el
        else:
            self.beam_el += max_d_el * (1.0 if d_el > 0 else -1.0)

        # Check if beam is on target
        rem_az = abs(normalize_angle(target_az - self.beam_az))
        rem_el = abs(target_el - self.beam_el)
        is_aligned = (rem_az <= self.beamwidth * 0.7) and (
            rem_el <= self.beamwidth * 0.7
        )
        return is_aligned

    def calculate_snr(self, range_m, rcs_m2, off_boresight_rad, uncertainty_m=0.0):
        """
        Computes received Signal-to-Noise Ratio (SNR in dB) using the Radar Range Equation.
        Accounts for target position uncertainty: larger uncertainty increases pointing error.
        """
        if range_m < self.min_range:
            return -99.0

        # R^4 range factor: 10 * log10((R_ref / R)^4) = 40 * log10(R_ref / R)
        range_loss_db = 40.0 * math.log10(self.r_ref / max(10.0, range_m))

        # RCS factor: 10 * log10(sigma / sigma_ref)
        rcs_gain_db = 10.0 * math.log10(max(0.01, rcs_m2) / self.rcs_ref)

        # Antenna beam pattern loss: 2-way Gaussian pattern with realistic sidelobe floor (-45 dB)
        # Position uncertainty induces an angular dispersion pointing error
        ang_uncertainty = math.atan2(max(0.0, uncertainty_m), max(20.0, range_m))
        effective_off = math.hypot(off_boresight_rad, ang_uncertainty)
        norm_off = effective_off / self.beamwidth
        pattern_loss_db = max(-45.0, -12.0 * (norm_off**2))

        # Atmospheric attenuation (~ 0.008 dB / km for X-band)
        atmo_loss_db = -0.008 * (range_m / 1000.0)

        total_snr_db = (
            self.snr_ref_db
            + range_loss_db
            + rcs_gain_db
            + pattern_loss_db
            + atmo_loss_db
        )
        return max(-99.0, total_snr_db)

    def calculate_detection_depth(self, rcs_m2):
        """
        Computes the maximum effective Detection Depth R_max for a target with given RCS.
        R_max = R_ref * (sigma / sigma_ref)^(1/4) * 10^((SNR_ref - SNR_thresh) / 40)
        """
        rcs_factor = (max(0.01, rcs_m2) / self.rcs_ref) ** 0.25
        margin_factor = 10.0 ** ((self.snr_ref_db - self.snr_threshold_db) / 40.0)
        r_max = self.r_ref * rcs_factor * margin_factor
        return min(self.max_nominal_depth * 1.3, r_max)

    def calculate_detection_probability(self, snr_db):
        """
        Converts SNR (dB) to detection probability Pd using a standard sigmoidal ROC curve.
        Safely bounded against floating-point overflow.
        """
        snr_diff = snr_db - self.snr_threshold_db
        exponent = -0.55 * snr_diff
        if exponent > 50.0:
            return 0.001
        elif exponent < -50.0:
            return 0.999
        pd = 1.0 / (1.0 + math.exp(exponent))
        return max(0.001, min(0.999, pd))

    def get_beam_cone_mesh(self, origin, depth=None):
        """
        Generates 3D coordinates for the volumetric beam cone:
        - Apex at carrier radar position
        - Circular/elliptical perimeter at range = depth
        Returns (apex, perimeter_points) in 3D world coordinates.
        """
        if depth is None:
            depth = self.current_detection_depth

        apex = origin
        num_segments = 16
        bw = self.beamwidth / 2.0
        cone_radius = depth * math.sin(bw)

        # Central boresight vector
        horiz = depth * math.cos(self.beam_el)
        bx = horiz * math.cos(self.beam_az)
        by = horiz * math.sin(self.beam_az)
        bz = depth * math.sin(self.beam_el)
        center_pt = (apex[0] + bx, apex[1] + by, apex[2] + bz)

        # Build orthonormal basis perpendicular to boresight
        b_vec = normalize3((bx, by, bz))
        up = (0.0, 0.0, 1.0) if abs(b_vec[2]) < 0.95 else (1.0, 0.0, 0.0)
        u_axis = normalize3(cross3(b_vec, up))
        v_axis = normalize3(cross3(u_axis, b_vec))

        perimeter = []
        for i in range(num_segments):
            angle = (2.0 * math.pi * i) / num_segments
            cu = cone_radius * math.cos(angle)
            cv = cone_radius * math.sin(angle)
            px = center_pt[0] + u_axis[0] * cu + v_axis[0] * cv
            py = center_pt[1] + u_axis[1] * cu + v_axis[1] * cv
            pz = center_pt[2] + u_axis[2] * cu + v_axis[2] * cv
            perimeter.append((px, py, pz))

        return apex, perimeter
