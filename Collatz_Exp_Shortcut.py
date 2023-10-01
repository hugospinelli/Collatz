import enum
import time

import numba as nb
import numpy as np
import matplotlib
import PIL


# Amount of times to print the total progress
PROGRESS_STEPS: int = 20

# Number of pixels (width*height) and aspect ratio (width/height)
RESOLUTION: int = 1920*1080
ASPECT_RATIO: float = 1920/1080

# Value of the center pixel
CENTER: complex = 0 + 0j  # For testing: -4.6875 + 2.63671875j

# Value range of the real part (width of the horizontal axis)
RE_RANGE: float = 10  # For testing: 10/16

# Show grid lines for integer real and imaginary parts
SHOW_GRID: bool = False
GRID_COLOR: tuple[int] = (125, 125, 125)

# Matplotlib named colormap
COLORMAP_NAME = 'inferno'

# Color of the interior of the fractal (convergent points)
INTERIOR_COLOR: tuple[int] = (0, 0, 60)
# Color for large divergence counts
ERROR_COLOR: tuple[int] = (0, 0, 60)


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


# Maximum iterations for the divergence test
MAX_ITER: int = 10**4  # recommended >= 10**3
# Minimum consecutive abs(r) decreases to declare linear divergence
MIN_R_DROPS: int = 4  # recommended >= 2
# Minimum iterations to start checking for slow drift (unknown divergence)
MIN_ITER_SLOW: int = 200  # recommended >= 100


# Max value of Re(z) and Im(z) such that the recursion doesn't overflow
CUTOFF_RE = 7.564545572282618e+153
CUTOFF_IM = 112.10398935569289


# Precompute the colormap
CMAP_LEN: int = 2000
cmap_mpl = matplotlib.colormaps[COLORMAP_NAME]
# Start away from 0 (discard black values for the 'inferno' colormap)
# Matplotlib's colormaps have 256 discrete color points
n_cmap = round(256*0.85)
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
    LINEAR = 3  # Detected linear divergence
    CUTOFF_RE = 4  # Diverged by exceeding the real part cutoff
    CUTOFF_IM = 5  # Diverged by exceeding the imaginary part cutoff


@nb.jit(nb.float64(nb.float64, nb.int64), nopython=True)
def smooth(x, k=1):
    y = np.expm1(np.pi*x)/np.expm1(np.pi)
    if k <= 1:
        return y
    return smooth(y, np.fmin(6, k - 1))


@nb.jit(nb.float64(nb.float64), nopython=True)
def get_delta_im(x):
    nu = np.log(np.abs(x)/CUTOFF_IM) / (np.pi*CUTOFF_IM - np.log(CUTOFF_IM))
    nu = np.fmax(0, np.fmin(nu, 1))
    return smooth(1 - nu, 2)


@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True)
def get_delta_re(x, e):
    nu = np.log(np.abs(x)/CUTOFF_RE) / np.log1p(e)
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

    cycle = 0
    r0 = -1
    r_drops = 0  # Counter for number of consecutive times abs(r) decreases
    delta_im = -1
    delta_re = -1
    a, b = z.real, z.imag
    a_cycle, b_cycle = a, b
    cutoff_re_squared = CUTOFF_RE*CUTOFF_RE
    
    for k in range(MAX_ITER):
        
        e = 0.5*np.exp(-np.pi*b)

        cycle += 1
        if cycle == 8:
            cycle = 0
            r = e*np.hypot(0.5 + a, b)/(1e-6 + np.abs(b))
            
            if r < r0 < 0.5:
                r_drops += 1
            else:
                r_drops = 0
            # Stop early due to likely slow linear divergence
            if r_drops >= MIN_R_DROPS:
                delta = 0.25*(CUTOFF_RE - a)
                return k + delta, DivType.LINEAR
            
            # Detected slow growth (maximum iterations will be reached)
            if ((k >= MIN_ITER_SLOW) and (r0 <= r)
                    and (r + (r - r0)*(MAX_ITER - k) < 8*0.05)):
                delta = 0.25*(CUTOFF_RE - a)
                return k + delta, DivType.SLOW
            r0 = r
            
            # Cycled back to the same value after 8 iterations
            if ((a - a_cycle)**2 + (b - b_cycle)**2 < 1e-16):
                return k, DivType.CYCLE
            a_cycle = a
            b_cycle = b

        a0 = a
        b0 = b
        s = np.sin(np.pi*a)
        c = np.cos(np.pi*a)
        # Equivalent to:
        # z = 0.25 + z - (0.25 + 0.5*z)*np.exp(np.pi*z*1j)
        # where z = a + b*1j
        a += e*(b*s - (0.5 + a)*c) + 0.25
        b -= e*(b*c + (0.5 + a0)*s)
        
        if b < -CUTOFF_IM:
            delta_im = get_delta_im(-b)
        if a*a + b*b > cutoff_re_squared:
            delta_re = get_delta_re(np.hypot(a, b), e)
        # Diverged by exceeding a cutoff
        if delta_im >= 0 or delta_re >= 0:
            if delta_re < 0 or delta_im <= delta_re:
                return k + delta_im, DivType.CUTOFF_IM
            else:
                return k + delta_re, DivType.CUTOFF_RE

    # Maximum iterations reached
    return k, DivType.MAX_ITER


@nb.jit(nb.complex128(nb.float64, nb.float64), nopython=True)
def pixel_to_z(a, b):
    re = X_MIN + (X_MAX - X_MIN)*a/WIDTH
    im = Y_MAX - (Y_MAX - Y_MIN)*b/HEIGHT
    return re + 1j*im


@nb.jit(nb.float64(nb.float64), nopython=True)
def cyclic_map(g):
    """A continuous function that cycles back and forth between 0 and 1."""
    # This can be any continuous function.
    # Log scale removes high-frequency color cycles.
    #g = np.log1p(np.fmax(0, g/freq_div)) - np.log1p(1/freq_div)

    # Normalize and cycle
    #g += 0.5  # phase from 0 to 1
    
    # Beyond this value for float64, decimals are truncated
    if g > 2251799813685247:
        return -1
    return 1 - np.abs(2*(g - np.floor(g)) - 1)


def get_pixel(px, py):
    z = pixel_to_z(px, py)
    dc, div_type = divergence_count(z)
    match div_type:
        case DivType.MAX_ITER | DivType.SLOW | DivType.LINEAR:
            # Default to background color for slow or unknown divergence
            return ERROR_COLOR
        case DivType.CYCLE:
            # Color of the interior of the fractal
            return INTERIOR_COLOR
        case DivType.CUTOFF_RE | DivType.CUTOFF_IM:
            cm = cyclic_map(dc)
            if 0 <= cm <= 1:
                return CMAP[round(cm*(CMAP_LEN - 1))]
            else:
                return ERROR_COLOR
        case _:
            # Placeholder for new types
            return ERROR_COLOR


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
    image = PIL.Image.new('RGB', (WIDTH, HEIGHT))
    pixels = image.load()

    prog = Progress(WIDTH, steps=PROGRESS_STEPS)
    for px in range(WIDTH):
        for py in range(HEIGHT):
            pixels[px, py] = get_pixel(px, py)
        if prog.check():
            print(f'{prog.progress:<7.1%}')

    if SHOW_GRID:
        for x in np.arange(np.ceil(X_MIN), np.floor(X_MAX) + 1):
            px = round((x - X_MIN)*(WIDTH - 1)/(X_MAX - X_MIN))
            for py in range(HEIGHT):
                pixels[px, py] = GRID_COLOR
        for y in np.arange(np.ceil(Y_MIN), np.floor(Y_MAX) + 1):
            py = round((Y_MAX - y)*(HEIGHT - 1)/(Y_MAX - Y_MIN))
            for px in range(WIDTH):
                pixels[px, py] = GRID_COLOR
    
    return image


image = create_image()
strtime = time.strftime('%Y%m%d-%H%M%S')
filename = f'Collatz_Exp_Shortcut_{strtime}.png'
image.save(filename)
