import itertools
import sys
import time
from math import pi, sqrt, log, exp, floor, ceil
from cmath import exp as cexp

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
# Warning: not optimized!
SHORTCUT = False

# Matplotlib named colormap
COLORMAP_NAME = 'inferno'

# Value of the center pixel
CENTER = 0 + 0j

# Value range of the real part (width of the horizontal axis)
RE_RANGE = 10

# Show grid lines for integer real and imaginary parts
SHOW_GRID = False
GRID_COLOR = (125, 125, 125)

# Color of the interior of the fractal (convergent points)
INTERIOR_COLOR = (0, 0, 60)
# Color for large divergence counts
ERROR_COLOR = (0, 0, 60)

# Plot range of the axes
X_MIN = CENTER.real - RE_RANGE/2  # min Re(z)
X_MAX = CENTER.real + RE_RANGE/2  # max Re(z)
Y_MIN = CENTER.imag - RE_RANGE/(2*ASPECT_RATIO)  # min Im(z)
Y_MAX = CENTER.imag + RE_RANGE/(2*ASPECT_RATIO)  # max Im(z)

WIDTH = round((X_MAX - X_MIN)*RESOLUTION/(Y_MAX - Y_MIN))
HEIGHT = round(RESOLUTION)


# Maximum iterations for the divergence test (recommended >= 10**3)
MAX_ITER = 2000


# Max value of Re(z) and Im(z) for which the recursion doesn't overflow
RE_CUTOFF = 7.564545572282618e+153
IM_CUTOFF = 112.10398935569289 if SHORTCUT else 111.95836403625282


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


#@nb.jit(nb.float64(nb.float64), nopython=True)
#def squeeze(x):
#    return sqrt(1 - (1-x)*(1-x))


@nb.jit(nb.float64(nb.float64, nb.int64), nopython=True)
def smooth(x, k=1):
    b = pi
    y = (exp(b*x) - 1)/(exp(b) - 1)
    if k <= 1:
        return y
    return smooth(y, min(10, k - 1))


@nb.jit(nb.float64(nb.float64), nopython=True)
def get_delta_im(x):
    nu = log(abs(x)/IM_CUTOFF)/(pi*IM_CUTOFF - log(IM_CUTOFF))
    nu = max(0, min(nu, 1))
    return smooth(1 - nu, 2)


@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True)
def get_delta_re(x, b):
    if SHORTCUT:
        nu = log(abs(x)/RE_CUTOFF)/(log(1 + 0.5*exp(-pi*b)))
    else:  # Regular
        nu = log(abs(x)/RE_CUTOFF)/(log(1.75 + 1.25*exp(-pi*b)))
    nu = max(0, min(nu, 1))
    return 1 - nu


@nb.jit(nb.types.containers.UniTuple(nb.float64, 2)(nb.complex128),
        nopython=True)
def divergence_count(z):
    max_cycle = 20
    cycle = 0
    z_cycle = z
    delta = -1
    for k in range(MAX_ITER):
        z0 = z
        if SHORTCUT:
            z = 0.25 + z - (0.25 + 0.5*z)*cexp(pi*z*1j)
            if exp(pi*z.imag) > abs(0.25 + 0.5*z)/(0.25*1e-3):
                delta = 0.25*(RE_CUTOFF - z.real)
                return k + delta, 0
        else:  # Regular
            z = 0.5 + 1.75*z - (0.5 + 1.25*z)*cexp(pi*z*1j)
            if z.imag > 12:  # 12 approx. -log(1e-16)/pi
                if abs(z + 2/3) > 1e-10:  # 2/3 = 0.5/(1.75 - 1)
                    delta = log(RE_CUTOFF/abs(z + 2/3))/log(1.75)
                    return k + delta, 0
        
        if -z.imag > IM_CUTOFF:
            delta = get_delta_im(-z.imag)
        if abs(z) > RE_CUTOFF:
            delta = max(delta, get_delta_re(abs(z), z0.imag))
        if delta >= 0:
            return k + delta, 0

        if z == z_cycle:
            return -k, cycle
        cycle += 1
        if cycle > max_cycle:
            cycle = 0
            z_cycle = z
    return -k, 0


@nb.jit(nb.float64(nb.float64), nopython=True)
def cyclic_map(g):
    """A continuous function that cycles back and forth from 0 to 1."""
    freq_div = 1 if SHORTCUT else 45
    # This can be any continuous function.
    # Log scale removes high-frequency color cycles.
    g = (g - 1)/freq_div
    #g = log(1 + max(0, g/freq_div)) - log(1 + 1/freq_div)

    # Normalize and cycle
    #g += 0.5  # phase from 0 to 1
    return 1 - abs(2*(g - floor(g)) - 1)


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
        g, cycle = divergence_count(c)
        if g >= 0:
            try:
                pix[p] = CMAP[round(cyclic_map(g)*(CMAP_LEN - 1))]
            # Value too large for numba's floor implementation
            except IndexError:
                #pix[p] = ERROR_COLOR
                if g < 1e50:
                    pix[p] = CMAP[round(np.random.random()*(CMAP_LEN - 1))]
                else:
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
