import itertools
import sys
import time
from math import pi, sqrt, log, floor, exp
from cmath import cos as ccos

import numba as nb
import numpy as np
import matplotlib
from PIL import Image
from PIL import ImageColor


# Amount of times to print the total progress
PROGRESS_STEPS = 20

# Width of the image in pixels and aspect ratio
RESOLUTION = 1080
ASPECT_RATIO = 1920/1080

# Set to `True` to plot the shortcut version of the fractal
SHORTCUT = False

# Matplotlib named colormap
COLORMAP_NAME = 'inferno'

# Value of the center pixel
CENTER = 0 + 0j

# Value range of the real part (width of the horizontal axis)
RE_RANGE = 5

# Plot range of the axes
X_MIN = CENTER.real - RE_RANGE/2  # min Re(z)
X_MAX = CENTER.real + RE_RANGE/2  # max Re(z)
Y_MIN = CENTER.imag - RE_RANGE/(2*ASPECT_RATIO)  # min Im(z)
Y_MAX = CENTER.imag + RE_RANGE/(2*ASPECT_RATIO)  # max Im(z)

WIDTH = round((X_MAX - X_MIN)*RESOLUTION/(Y_MAX - Y_MIN))
HEIGHT = round(RESOLUTION)


# Maximum iterations for the divergence test (recommended >= 60)
MAX_ITER = 60


# Max value of Re(z) and Im(z) for which the recursion doesn't overflow
RE_CUTOFF = 7.564545572282618e+153
IM_CUTOFF = 112.10398935569289 if SHORTCUT else 111.95836403625282

# Smallest positive real fixed point
INNER_FIXED_POINT = 0.277733766171606 if SHORTCUT else 0.150108511304474


# Precompute the colormap
CMAP_LEN = 2000
cmap_mpl = matplotlib.colormaps[COLORMAP_NAME]
n_cmap = 256
# Start away from 0 (discard black values for the 'inferno' colormap)
# Matplotlib's colormaps have 256 discrete color points
n_cmap = round(0.98*n_cmap)
CMAP = [cmap_mpl(k/256) for k in range(256 - n_cmap, 256)]
# Interpolate
x = np.linspace(0, 1, num=CMAP_LEN)
xp = np.linspace(0, 1, num=n_cmap)
c0, c1, c2 = tuple(np.interp(x, xp, [c[k] for c in CMAP]) for k in range(3))
CMAP = []
for x0, x1, x2 in zip(c0, c1, c2):
    CMAP.append(tuple(round(255*x) for x in (x0, x1, x2)))


@nb.jit(nb.float64(nb.float64, nb.int64), nopython=True)
def smooth(x, k=1):
    y = (exp(pi*x) - 1)/(exp(pi) - 1)
    if k <= 1:
        return y
    return smooth(y, min(6, k-1))


@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True)
def get_delta(x, cutoff):
    nu = log(abs(x)/cutoff)/(pi*cutoff - log(cutoff))
    nu = max(0, min(nu, 1))
    return smooth(1 - nu, 2)


@nb.jit(nb.float64(nb.complex128), nopython=True)
def growth_rate(z):
    for k in range(MAX_ITER):
        if SHORTCUT:
            z = 0.25 + z - (0.25 + 0.5*z)*ccos(pi*z)
        else:  # Regular
            z = 0.5 + 1.75*z - (0.5 + 1.25*z)*ccos(pi*z)
        
        if abs(z.imag) > IM_CUTOFF:
            return k + get_delta(z.imag, IM_CUTOFF)
        if abs(z.real) > RE_CUTOFF:
            return k + get_delta(z.real, RE_CUTOFF)
        if abs(z) < INNER_FIXED_POINT:
            return -1
    return -1


@nb.jit(nb.float64(nb.float64), nopython=True)
def cyclic_map(g: float) -> float:
    """A continuous function that cycles back and forth from 0 to 1."""
    # This can be any continuous function.
    # Log scale removes high-frequency color cycles.
    g = log(1 + max(0, (g - 1)/12))

    # Normalize and cycle
    x = 2*(g - floor(g))
    if x > 1:
        return 2 - x
    return x


@nb.jit(nb.complex128(nb.types.containers.UniTuple(nb.float64, 2)),
        nopython=True)
def pixel_to_z(p):
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
    n_pix = WIDTH*HEIGHT

    prog = Progress(n_pix, steps=PROGRESS_STEPS)
    for p in itertools.product(range(WIDTH), range(HEIGHT)):
        c = pixel_to_z(p)
        g = growth_rate(c)
        if g >= 0:
            pix[p] = CMAP[round(cyclic_map(g)*(CMAP_LEN - 1))]
        else:
            # Color of the interior of the fractal
            pix[p] = (0, 0, 0)
        if prog.check():
            print(f'{prog.progress:<7.1%}')
    return img


img = create_image()
strtime = time.strftime('%Y%m%d-%H%M%S')
fractal_type = 'Shortcut' if SHORTCUT else 'Regular'
filename = f'Collatz_{fractal_type}_{strtime}.png'
img.save(filename)
