from .core_tile import CoreTile, SimpleTile, QuadrantsTile
from .core_form_factors import CoreFormFactors
from .._scarabee import DiffusionGeometry, NEM4DiffusionDriver
import numpy as np


class CoreBuilder:
    """
    The CoreBuilder class facilitates the construction of a NEM4DiffusionDriver
    instance and a CoreFormFactors instnace, which are used to perform a full
    core simulation and compute assembly/pin powers. This class not not yet
    fully support defining cores with quarter symmetry in the x-y plane. Users
    should perform full core nodal simulations for the time being.

    The user provided axial discretization is directly used in the nodal
    simulation, and will not be futher discretized.

    Parameters
    ----------
    tile_width : foat
        Width / pitch of an assembly in the core.
    num_tiles : int
        Number of tiles along the diameter of the simulation. This is likely
        the number of assemblies across the core plus two (for a reflector tile
        on each side).
    pitch : float
        The fuel pin pitch within an assembly.
    num_pins : int
        Number of fuel pins along one side of an assembly.
    tiles : np.ndarray of CoreTile or float
        A 3D array describing the geometry of the core. The array is indexed
        along z, y, then x, with z and y being indexed from large to small
        values (which permits a "what you see is what you get in your editor).
        Each entry in tiles should either be a SimpleTile or a QuadrantsTile.
        An entry of 0. can be used indicate a tile which is not used.
    z_widths : np.ndarray
        A 2D array for the axial height of each axial slice of the geometry.
        This is the axial discretization which will be used in the nodal
        simulation.
    zmin_albedo : float
        Albedo boundary condition at the -z boundary.
    zmax_albedo : float
        Albedo boundary condition at the +z boundary.
    """

    def __init__(
        self,
        tile_width: float,
        num_tiles: int,
        pitch: float,
        num_pins: int,
        tiles: np.ndarray,
        z_widths: np.ndarray,
        zmin_albedo: float,
        zmax_albedo: float,
    ):
        if tile_width <= 0.0:
            raise ValueError("Tile width must be > 0.")
        if num_tiles <= 0.0:
            raise ValueError("Number of tiles must be > 0.")
        if num_pins <= 0:
            raise ValueError("The number of pins per assembly must be > 0.")
        if pitch <= 0.0:
            raise ValueError("The pin pitch must be > 0.")
        if tiles.ndim != 3:
            raise ValueError("Array of tiles must be 3D array of CoreTile instances.")
        if z_widths.ndim != 1 or z_widths.size != tiles.shape[0]:
            raise ValueError(
                "Array of z_widths must be 1D array with same length as tiles.shape[0]."
            )

        twice_gap_width = tile_width - num_pins * pitch
        if twice_gap_width < -1.0e-6:
            raise ValueError(
                "The number of pins per assembly and the pin pitch are not compatible with the tile width."
            )
        elif twice_gap_width < 0.0:
            twice_gap_width = 0.0

        self.tile_width = tile_width
        self.num_tiles = num_tiles
        self.num_pins = num_pins
        self.gap_width = twice_gap_width / 2.0
        self.pitch = pitch
        self.xmin_albedo = 0.0
        self.xmax_albedo = 0.0
        self.ymin_albedo = 0.0
        self.ymax_albedo = 0.0
        self.zmin_albedo = zmin_albedo
        self.zmax_albedo = zmax_albedo
        self.z_widths = z_widths

        # The constructor is not quite quadrant symmetry compatible yet

        # Build the core form factors
        ff_tiles = []
        for k in range(tiles.shape[0]):
            ff_tiles.append([])
            for j in range(tiles.shape[1]):
                ff_tiles[-1].append([])
                for i in range(tiles.shape[2]):
                    if isinstance(tiles[k, j, i], CoreTile):
                        ff_tiles[-1][-1].append(tiles[k, j, i].form_factors)
                    else:
                        ff_tiles[-1][-1].append(0.0)
        self.core_form_factors = CoreFormFactors(
            tile_width, num_tiles, np.array(ff_tiles), self.z_widths
        )

        # Get all the x and y indices which have double tiles
        double_x_inds = set()
        double_y_inds = set()
        for k in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                for i in range(tiles.shape[2]):
                    tile = tiles[k, j, i]

                    if not isinstance(tile, CoreTile):
                        continue

                    if tile.num_x_slots == 2:
                        double_x_inds.add(i)
                    if tile.num_y_slots == 2:
                        double_y_inds.add(j)

        # Get widths arrays along each dimension from the form factors
        self.dz = z_widths.copy()
        self.nz = np.ones(self.dz.size, dtype=np.int64)
        self.dy = []
        self.dx = []
        self.nx = []
        self.ny = []

        for j in range(tiles.shape[1]):
            if j in double_y_inds:
                self.dy.append(0.5 * self.tile_width)
                self.dy.append(0.5 * self.tile_width)
                self.ny.append(1)
                self.ny.append(1)
            else:
                self.dy.append(self.tile_width)
                self.ny.append(2)
        # Must flip y, as they go backwards in tiles
        self.dy = np.flip(np.array(self.dy))
        self.ny = np.flip(np.array(self.ny, dtype=np.int64))

        for i in range(tiles.shape[2]):
            if i in double_x_inds:
                self.dx.append(0.5 * self.tile_width)
                self.dx.append(0.5 * self.tile_width)
                self.nx.append(1)
                self.nx.append(1)
            else:
                self.dx.append(self.tile_width)
                self.nx.append(2)
        # No need to flip x, as they go in order !
        self.dx = np.array(self.dx)
        self.nx = np.array(self.nx)

        # Now we build the NEM simulation. We use 2 nodes per assembly
        # radially, and we model them explicitly instead of using the implicit
        # node division which is provided by the DiffusionGeometry class.
        self.geometry_tiles = []

        def add_tile(tile, second_x, second_y):
            if isinstance(tile, SimpleTile):
                self.geometry_tiles.append(tile.diffusion_data)
            elif isinstance(tile, QuadrantsTile):
                if second_x and not second_y:
                    self.geometry_tiles.append(tile.quad1)
                elif not second_x and not second_y:
                    self.geometry_tiles.append(tile.quad2)
                elif not second_x and second_y:
                    self.geometry_tiles.append(tile.quad3)
                elif second_x and second_y:
                    self.geometry_tiles.append(tile.quad4)
            elif isinstance(tile, int) or isinstance(tile, float):
                if tile < 0.0:
                    raise ValueError(f"Found scaral tile with value of {tile}")
                self.geometry_tiles.append(float(tile))
            else:
                raise TypeError(f"Unknown core tile type {type(tile)}.")

        # Tile indices
        for k in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                second_y = False
                for i in range(tiles.shape[2]):
                    tile = tiles[k, j, i]
                    add_tile(tile, False, second_y)
                    if i in double_x_inds:
                        add_tile(tile, True, second_y)

                if j in double_y_inds:
                    second_y = True
                    for i in range(tiles.shape[2]):
                        tile = tiles[k, j, i]
                        add_tile(tile, False, second_y)
                        if i in double_x_inds:
                            add_tile(tile, True, second_y)

        # Tiles have been created. Now we build the geometry
        self.diffusion_geometry = DiffusionGeometry(
            self.geometry_tiles,
            self.dx,
            self.nx,
            self.dy,
            self.ny,
            self.dz,
            self.nz,
            self.xmin_albedo,
            self.xmax_albedo,
            self.ymin_albedo,
            self.ymax_albedo,
            self.zmin_albedo,
            self.zmax_albedo,
        )
        self.solver = NEM4DiffusionDriver(self.diffusion_geometry)

        # For full core simulations, better to set a flux tolerance of 1.E-6 to
        # avoid asymmetric errors across the core
        self.solver.flux_tolerance = 1.0e-6

        # Temporary so we know that these types exist. Are filled by the
        # _compute_pin_centers method below.
        self.x_pin_centers = np.zeros(self.num_tiles * self.num_pins)
        self.y_pin_centers = np.zeros(self.num_tiles * self.num_pins)

        # We should now compute the pin centers
        self._compute_pin_centers()

    def _compute_pin_centers(self):
        # This method is NOT Quadrant symmetry compatible yet !

        # Compute coordinate of center point of each pin
        # Add gap for left side of first assembly, plus half a pitch to get
        # to the center of first pin
        x_coord = self.gap_width + 0.5 * self.pitch
        # Next, we subtract one pitch so that we can add a pitch for each iteration of the loop
        x_coord -= self.pitch
        i = 0
        for A in range(self.num_tiles):
            for p in range(self.num_pins):
                x_coord += self.pitch
                self.x_pin_centers[i] = x_coord
                i += 1
            # At end of an assembly, must move over two gap widths
            x_coord += 2.0 * self.gap_width

        # The y points should be identical
        self.y_pin_centers = self.x_pin_centers.copy()

    def compute_assembly_powers(self) -> np.ndarray:
        """
        Computes the axially integrated average assembly powers. The powers are
        normalized such that the average power is unity, only considering tiles
        with a non-zero power.

        Returns
        -------
        np.ndarray
            A 2D Numpy array with the average assembly powers.
        """
        # This method is NOT Quadrant symmetry compatible yet !

        # First, get the node powers
        node_powers = self.solver.avg_power()

        # Sum node powers axially
        node_powers = np.sum(node_powers, axis=2)
        node_powers /= np.mean(node_powers[np.where(node_powers != 0.0)])

        asmbly_powers = np.zeros((self.num_tiles, self.num_tiles))
        N = 2
        for i in range(self.num_tiles):
            for j in range(self.num_tiles):
                asmbly_powers[i, j] = np.sum(
                    node_powers[i * N : i * N + N, j * N : j * N + N]
                )
        asmbly_powers /= float(N * N)

        return asmbly_powers

    def compute_pin_powers(self, z: float | np.ndarray) -> np.ndarray:
        """
        Computes the relative pin powers such that the average of all pin
        powers is unity, only considering pins with a non-zero power.

        Parameters
        ----------
        z : float or np.ndarray
            The z coordinates at which to evaluate the pin powers.

        Returns
        -------
        np.ndarray
            A 3D Numpy array with the pin powers evaluated at the pin centers
            and the provided z points. Indexed according to x, y, and z.
        """
        # This method is NOT Quadrant symmetry compatible yet !

        if isinstance(z, float):
            z = np.array([z])
        if not isinstance(z, np.ndarray) or z.ndim != 1:
            raise TypeError("z must be a 1D Numpy array.")

        x = self.x_pin_centers
        y = self.y_pin_centers

        # We now evaluate the form factors at the provided positions
        form_factors = self.core_form_factors(x, y, z)

        # Next, we compute the homogeneous power distribution. We must
        # average the offset of 4 points, to accound for pins split by the
        # node boundaries, otherwise we might have odd "cusping" effects.
        homog_power = self.solver.power(x + 0.25 * self.pitch, y + 0.25 * self.pitch, z)
        homog_power += self.solver.power(
            x - 0.25 * self.pitch, y + 0.25 * self.pitch, z
        )
        homog_power += self.solver.power(
            x - 0.25 * self.pitch, y - 0.25 * self.pitch, z
        )
        homog_power += self.solver.power(
            x + 0.25 * self.pitch, y - 0.25 * self.pitch, z
        )
        homog_power /= 4.0

        pin_powers = form_factors * homog_power

        # We average based on the non-zero entries
        nmsk = np.where(pin_powers != 0.0)
        avg = np.mean(pin_powers[nmsk])
        pin_powers /= avg

        return pin_powers
