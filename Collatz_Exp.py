import enum
import itertools
import time
from math import floor, ceil

import numba as nb
import numpy as np
import matplotlib
from PIL import Image, PyAccess


# Amount of times to print the total progress
PROGRESS_STEPS: int = 20

# Number of pixels (width*height) and aspect ratio (width/height)
RESOLUTION: int = 1920*1080
ASPECT_RATIO: float = 1920/1080

# Value of the center pixel
CENTER: complex = 0 + 0j

# Value range of the real part (width of the horizontal axis)
RE_RANGE: float = 10

# Set to `True` to plot the shortcut version of the fractal
# Warning: not optimized!
SHORTCUT: bool = False

# Matplotlib named colormap
COLORMAP_NAME: str = 'inferno'

# Show grid lines for integer real and imaginary parts
SHOW_GRID: bool = False
GRID_COLOR: tuple[int, int, int] = (125, 125, 125)

# Color of the interior of the fractal (convergent points)
INTERIOR_COLOR: tuple[int, int, int] = (0, 0, 60)
# Color for large divergence counts
ERROR_COLOR: tuple[int, int, int] = (0, 0, 60)

# Plot range of the axes
X_MIN = CENTER.real - RE_RANGE/2  # min Re(z)
X_MAX = CENTER.real + RE_RANGE/2  # max Re(z)
Y_MIN = CENTER.imag - RE_RANGE/(2*ASPECT_RATIO)  # min Im(z)
Y_MAX = CENTER.imag + RE_RANGE/(2*ASPECT_RATIO)  # max Im(z)

x_range = X_MAX - X_MIN
y_range = Y_MAX - Y_MIN
pixels_per_unit = np.sqrt(RESOLUTION/(x_range*y_range))

# Width and height of the image in pixels
WIDTH = round(pixels_per_unit*x_range)
HEIGHT = round(pixels_per_unit*y_range)


# Maximum iterations for the divergence test (recommended >= 10**3)
MAX_ITER = 2000


# Max value of Re(z) and Im(z) for which the recursion doesn't overflow
CUTOFF_RE = 7.564545572282618e+153
CUTOFF_IM = 112.10398935569289 if SHORTCUT else 111.95836403625282


# Precompute the colormap
CMAP_LEN = 2000
cmap_mpl = matplotlib.colormaps[COLORMAP_NAME]
# Start away from 0 (discard black values for the 'inferno' colormap)
# Matplotlib's colormaps have 256 discrete color points
n_cmap = round(256*(0.85 if SHORTCUT else 0.9))
CMAP = [cmap_mpl(k/256) for k in range(256 - n_cmap, 256)]
# Interpolate
x = np.linspace(0, 1, num=CMAP_LEN)
xp = np.linspace(0, 1, num=n_cmap)
c0, c1, c2 = tuple(np.interp(x, xp, [c[k] for c in CMAP]) for k in range(3))
CMAP = []
for x0, x1, x2 in zip(c0, c1, c2):
    CMAP.append(tuple(round(255*x) for x in (x0, x1, x2)))


class DivType(enum.Enum):
    """Divergence type."""

    MAX_ITER = 0  # Maximum iterations reached
    SLOW = 1  # Detected slow growth (maximum iterations will be reached)
    CYCLE = 2  # Cycled back to the same value after 8 iterations
    CUTOFF_RE = 3  # Diverged by exceeding the real part cutoff
    CUTOFF_IM = 4  # Diverged by exceeding the imaginary part cutoff


@nb.jit(nb.float64(nb.float64, nb.int64), nopython=True)
def smooth(x, k=1):
    """Recursive exponential smoothing function."""

    y = np.expm1(np.pi*x)/np.expm1(np.pi)
    if k <= 1:
        return y
    return smooth(y, np.fmin(10, k - 1))


@nb.jit(nb.float64(nb.float64), nopython=True)
def get_delta_im(x):
    """Get the fractional part of the smoothed divergence count for
    imaginary part blow-up."""

    nu = np.log(np.abs(x)/CUTOFF_IM) / (np.pi*CUTOFF_IM - np.log(CUTOFF_IM))
    nu = np.fmax(0, np.fmin(nu, 1))
    return smooth(1 - nu, 2)


@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True)
def get_delta_re(x, b):
    """Get the fractional part of the smoothed divergence count for
    real part blow-up."""

    if SHORTCUT:
        nu = np.log(np.abs(x)/CUTOFF_RE)/(np.log1p(0.5*np.exp(-np.pi*b)))
    else:  # Regular
        nu = np.log(np.abs(x)/CUTOFF_RE)/(np.log(1.75 + 1.25*np.exp(-np.pi*b)))
    nu = np.fmax(0, np.fmin(nu, 1))
    return 1 - nu


@nb.jit(
    nb.types.containers.Tuple((
        nb.float64,
        nb.types.EnumMember(DivType, nb.int64)
    ))(nb.complex128),
    nopython=True
)
def divergence_count(z):
    """Return a smoothed divergence count and the type of divergence."""

    delta_im = -1
    delta_re = -1
    max_cycle = 20
    cycle = 0
    z_cycle = z
    for k in range(MAX_ITER):
        z0 = z
        if SHORTCUT:
            z = 0.25 + z - (0.25 + 0.5*z)*np.exp(np.pi*z*1j)
            if np.exp(np.pi*z.imag) > np.abs(0.25 + 0.5*z)/(0.25*1e-3):
                delta = 0.25*(CUTOFF_RE - z.real)
                # Stop early due to likely slow divergence
                return k + delta, DivType.SLOW
        else:  # Regular
            z = 0.5 + 1.75*z - (0.5 + 1.25*z)*np.exp(np.pi*z*1j)
            if z.imag > 12:  # 12 approx. -np.log(1e-16)/pi
                if np.abs(z + 2/3) > 1e-10:  # 2/3 = 0.5/(1.75 - 1)
                    delta = np.log(CUTOFF_RE/np.abs(z + 2/3))/np.log(1.75)
                    # Stop early due to likely slow divergence
                    return k + delta, DivType.SLOW

        if -z.imag > CUTOFF_IM:
            delta_im = get_delta_im(-z.imag)
        if np.abs(z) > CUTOFF_RE:
            delta_re = get_delta_re(np.abs(z), z0.imag)
        # Diverged by exceeding a cutoff
        if delta_im >= 0 or delta_re >= 0:
            if delta_re < 0 or delta_im <= delta_re:
                return k + delta_im, DivType.CUTOFF_IM
            else:
                return k + delta_re, DivType.CUTOFF_RE

        if z == z_cycle:
            # Cycled back to the same value after `cycle` iterations
            return -k, DivType.CYCLE
        cycle += 1
        if cycle > max_cycle:
            cycle = 0
            z_cycle = z

    # Maximum iterations reached
    return -1, DivType.MAX_ITER


@nb.jit(nb.float64(nb.float64), nopython=True)
def cyclic_map(g):
    """A continuous function that cycles back and forth from 0 to 1."""

    # This can be any continuous function.
    # Log scale removes high-frequency color cycles.
    freq_div = 1 if SHORTCUT else 45
    g = (g - 1)/freq_div
    # g = np.log1p(np.fmax(0, g/freq_div)) - np.log1p(1/freq_div)

    # Beyond this value for float64, decimals are truncated
    if g >= 2**51:
        return -1

    # Normalize and cycle
    # g += 0.5  # phase from 0 to 1
    return 1 - np.abs(2*(g - np.floor(g)) - 1)


@nb.jit(nb.complex128(nb.types.containers.UniTuple(nb.float64, 2)),
        nopython=True)
def pixel_to_z(p):
    """Convert pixel coordinates to its corresponding complex value."""

    re = X_MIN + (X_MAX - X_MIN)*p[0]/WIDTH
    im = Y_MAX - (Y_MAX - Y_MIN)*p[1]/HEIGHT
    return re + 1j*im


class Progress:
    """Simple progress check helper class."""

    def __init__(self, n: int, steps: int = 10):
        self.n = n
        self.k = 0
        self.steps = steps
        self.step = 1
        self.progress = 0

    def check(self) -> bool:
        self.k += 1
        self.progress = self.k/self.n
        if self.steps*self.k >= self.step*self.n:
            self.step += 1
            return True
        return self.progress == 1


def create_image():
    img = Image.new('RGB', (WIDTH, HEIGHT))
    pix = img.load()
    pix: PyAccess
    n_pix = WIDTH*HEIGHT

    prog = Progress(n_pix, steps=PROGRESS_STEPS)
    for p in itertools.product(range(WIDTH), range(HEIGHT)):
        c = pixel_to_z(p)
        g, div_type = divergence_count(c)
        if g >= 0:
            dg = cyclic_map(g)
            if dg >= 0:
                pix[p] = CMAP[round(dg*(CMAP_LEN - 1))]
            else:
                # Value too large for numba's floor implementation
                pix[p] = ERROR_COLOR
        else:
            # Color of the interior of the fractal
            pix[p] = INTERIOR_COLOR
        if prog.check():
            print(f'{prog.progress:<7.1%}')

    if SHOW_GRID:
        for x in range(ceil(X_MIN), floor(X_MAX) + 1):
            px = round((x - X_MIN)*(WIDTH - 1)/(X_MAX - X_MIN))
            for py in range(HEIGHT):
                pix[(px, py)] = GRID_COLOR
        for y in range(ceil(Y_MIN), floor(Y_MAX) + 1):
            py = round((Y_MAX - y)*(HEIGHT - 1)/(Y_MAX - Y_MIN))
            for px in range(WIDTH):
                pix[(px, py)] = GRID_COLOR

    return img


img = create_image()
strtime = time.strftime('%Y%m%d-%H%M%S')
fractal_type = 'Shortcut' if SHORTCUT else 'Regular'
filename = f'Collatz_Exp_{fractal_type}_{strtime}.png'
img.save(filename)
