import numpy as np
import healpy as hp
import random as rand
from pathos.multiprocessing import ProcessPool

from typing import Tuple, List

from dipolefunctions_CatWISE import *

lon_CMB = 264.021
lat_CMB = 48.253
v_CMB = 369820.

def deg2ang(lon, lat):
    """
    Given longitude and latitude in degrees,
    return theta and phi in radians.
    """
    theta = (90. - lat) * np.pi / 180.
    phi = lon * np.pi / 180.
    return theta, phi

def ang2deg(theta, phi):
    """
    Given theta and phi in radians,
    return longitude and latitude in degrees.
    """ 
    lon = phi * 180. / np.pi
    lat = 90. - theta * 180. / np.pi
    return lon, lat

def dip2alm(mag: float, lon: float = lon_CMB,
        lat: float = lat_CMB, random: bool = False) -> np.ndarray:
    """
    Given a dipole magnitude, and direction in longitude and
    latitude in degrees, return the alm of the map with such
    a pure dipole.
    """
    theta, phi = (np.arccos(2*rand.random()-1), 2*np.pi*rand.random()) if random else deg2ang(lon, lat)
    a00 = np.sqrt(4*np.pi) + 0.0j
    a10 = mag * np.sqrt(4*np.pi/3) * np.cos(theta) + 0.0j
    a11 = mag * np.sqrt(2*np.pi/3) * np.sin(theta) * (-np.cos(phi) + 1j*np.sin(phi))
    alm = [a00, a10, a11]
    return np.array(alm)

def point_picking(weighting: np.ndarray, points: int,
        threads: int = 1) -> np.ndarray:
    """
    Distributes points according to weighting in a an array.
    Supports multithreading.
    """
    if (threads == 1):
        return point_picking_old(weighting, points)
    bins = range(len(weighting))
    grand_dist = rand.choices(bins, weighting, k=points)
    dist = np.array_split(grand_dist, threads)
    def hist_func(d: np.ndarray):
        h = np.zeros(len(weighting), dtype=int)
        for i in d:
            h[i] += 1
        return h
    with ProcessPool(nodes=threads) as pool:
        hist = np.array(list(pool.map(hist_func, dist)))
    return hist.sum(axis=0)

def point_picking_old(weighting: np.ndarray, points: int) -> np.ndarray:
    """
    Distributes points according to weighting in a an array.
    Does not support multithreading.
    """
    bins = range(len(weighting))
    dist = rand.choices(bins, weighting, k=points)
    hist = np.zeros(len(weighting), dtype=int)
    for d in dist:
        hist[d] += 1
    return hist

def dipole_lonlat(imap: np.ndarray,
        mode: str = 'dip') -> Tuple[float]:
    """
    Fit dipole to map, and return the relative dipole magnitude,
    along with the direction it is pointing at.
    """
    mono, dip = 0, 0
    if mode == 'dip':
        mono, dip = hp.fit_dipole(imap)
    elif mode == 'quad':
        mono, dip, quad = fit_quadrupole(imap)
    dipmag = np.sqrt(dip[0]**2 + dip[1]**2 + dip[2]**2)
    mag = dipmag / mono
    unitdip = dip / dipmag
    return mag, *vec2dir(unitdip)

def generate_sim(iters: int, dip: float, udip: float,
        nside: int, point: int, mask: np.ndarray,
        threads: int = 1, mode: str = 'dip') -> List[np.ndarray]:
    """
    Simulate distributions.
    """
    if dip == 0 and udip == 0:
        return generate_monosim(iters, nside, point, mask, threads, mode)

    normals = np.random.normal(dip, udip, iters)
    # alms = np.array(list(map(dip2alm, normals)))
    # def weights_func(alm):
    #     return hp.alm2map(alm, nside)
    # weights = np.array(list(map(weights_func, alms)))
    def sim_func(n: np.ndarray):
        alm = dip2alm(n, random=True)
        weight = hp.alm2map(alm, nside)
        dip_map = point_picking(weight, point)
        full_dips = dipole_lonlat(dip_map, mode)
        dip_mask = np.where(mask, dip_map, hp.UNSEEN)
        cut_dips = dipole_lonlat(dip_mask, mode)
        return *full_dips, *cut_dips
    with ProcessPool(nodes=threads) as pool:
        data = np.array(list(pool.map(sim_func, normals)))
    return list(zip(*data))

def generate_monosim(iters: int,
        nside: int, point: int, mask: np.ndarray,
        threads: int = 1, mode: str = 'dip') -> List[np.ndarray]:
    """
    Simulate distributions for monopoles.
    """
    normals = np.zeros(iters)
    def sim_func(n: np.ndarray):
        alm = np.array([1.+0.j, 0.+0.j, 0.+0.j])
        weight = hp.alm2map(alm, nside)
        dip_map = point_picking(weight, point)
        full_dips = dipole_lonlat(dip_map, mode)
        dip_mask = np.where(mask, dip_map, hp.UNSEEN)
        cut_dips = dipole_lonlat(dip_mask, mode)
        return *full_dips, *cut_dips
    with ProcessPool(nodes=threads) as pool:
        data = np.array(list(pool.map(sim_func, normals)))
    return list(zip(*data))

def generate_multisim(iters: int, dip: float, udip: float,
        nside: int, point: int, masks: np.ndarray,
        threads: int = 1, mode: str = 'dip') -> List[np.ndarray]:
    """
    Simulate distributions.
    """

    normals = np.random.normal(dip, udip, iters)
    # alms = np.array(list(map(dip2alm, normals)))
    # def weights_func(alm):
    #     return hp.alm2map(alm, nside)
    # weights = np.array(list(map(weights_func, alms)))
    def sim_func(n: np.ndarray):
        alm = dip2alm(n, random=True)
        weight = hp.alm2map(alm, nside)
        dip_map = point_picking(weight, point)
        full_dips = dipole_lonlat(dip_map, mode)
        cut_dips = []
        for m in masks:
            dip_mask = np.where(m, dip_map, hp.UNSEEN)
            cut_dips.append(dipole_lonlat(dip_mask, mode))
        return full_dips, *cut_dips
    with ProcessPool(nodes=threads) as pool:
        data = np.array(list(pool.map(sim_func, normals)))
    return list(zip(*data))

def sky_cov(mask: np.ndarray) -> float:
    return float(sum(mask)) / float(len(mask))

def fit_quadrupole(m, nest=False, bad=hp.UNSEEN, gal_cut=0):
    """Fit a quadrupole, a dipole, and a monopole to the map, excluding bad pixels.
    Parameters
    ----------
    m : float, array-like
      the map to which a dipole is fitted and subtracted, accepts masked maps
    nest : bool
      if ``False`` m is assumed in RING scheme, otherwise map is NESTED
    bad : float
      bad values of pixel, default to :const:`UNSEEN`.
    gal_cut : float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] degrees are not taken into account
    Returns
    -------
    res : tuple of length 2
      the monopole value in res[0] and the dipole vector (as array) in res[1:4] and the quadrupole matrix values are in res[4:] 
    See Also
    --------
    remove_dipole, fit_monopole, remove_monopole
    """
    m = hp.pixelfunc.ma_to_array(m)
    m = np.asarray(m)
    npix = m.size
    nside = hp.npix2nside(npix)
    if nside > 128:
        bunchsize = npix // 24
    else:
        bunchsize = npix
    aa = np.zeros((9, 9), dtype=np.float64)
    v = np.zeros(9, dtype=np.float64)
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(m.flat[ipix] != bad) & (np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, nest)
        if gal_cut > 0:
            w = np.abs(z) >= np.sin(gal_cut * np.pi / 180)
            ipix = ipix[w]
            x = x[w]
            y = y[w]
            z = z[w]
            del w
        aa[0, 0] += ipix.size
        aa[1, 0] += x.sum()
        aa[2, 0] += y.sum()
        aa[3, 0] += z.sum()
        aa[4, 0] += (0.5*(x**2-z**2)).sum()
        aa[5, 0] += (0.5*2*x*y).sum()
        aa[6, 0] += (0.5*2*x*z).sum()
        aa[7, 0] += (0.5*(y**2-z**2)).sum()
        aa[8, 0] += (0.5*2*y*z).sum()
        aa[1, 1] += (x**2).sum()
        aa[2, 1] += (x*y).sum()
        aa[3, 1] += (x*z).sum()
        aa[4, 1] += (0.5*x*(x**2-z**2)).sum()
        aa[5, 1] += (0.5*2*y*x**2).sum()
        aa[6, 1] += (0.5*2*z*x**2).sum()
        aa[7, 1] += (0.5*x*(y**2-z**2)).sum()
        aa[8, 1] += (0.5*2*x*y*z).sum()
        aa[2, 2] += (y**2).sum()
        aa[3, 2] += (y*z).sum()
        aa[4, 2] += (0.5*y*(x**2-z**2)).sum()
        aa[5, 2] += (0.5*2*x*y**2).sum()
        aa[6, 2] += (0.5*2*x*y*z).sum()
        aa[7, 2] += (0.5*y*(y**2-z**2)).sum()
        aa[8, 2] += (0.5*2*z*y**2).sum()
        aa[3, 3] += (z ** 2).sum()
        aa[4, 3] += (0.5*z*(x**2-z**2)).sum()
        aa[5, 3] += (0.5*2*x*y*z).sum()
        aa[6, 3] += (0.5*2*x*z**2).sum()
        aa[7, 3] += (0.5*x*y*(y**2-z**2)).sum()
        aa[8, 3] += (0.5*2*y*z**2).sum()
        aa[4, 4] += (0.5**2*(x**2-z**2)**2).sum()
        aa[5, 4] += (0.5**2*2*y*x*(x**2-z**2)).sum()
        aa[6, 4] += (0.5**2*2*z*x*(x**2-z**2)).sum()
        aa[7, 4] += (0.5**2*(x**2-z**2)*(y**2-z**2)).sum()
        aa[8, 4] += (0.5**2*2*y*z*(x**2-z**2)).sum()
        aa[5, 5] += (0.5**2*4*x**2*y**2).sum()
        aa[6, 5] += (0.5**2*4*y*z*x**2).sum()
        aa[7, 5] += (0.5**2*2*x*y*(y**2-z**2)).sum()
        aa[8, 5] += (0.5**2*4*x*z*y**2).sum()
        aa[6, 6] += (0.5**2*4*x**2*z**2).sum()
        aa[7, 6] += (0.5**2*2*x*z*(y**2-z**2)).sum()
        aa[8, 6] += (0.5**2*4*x*y*z**2).sum()
        aa[7, 7] += (0.5**2*(y**2-z**2)**2).sum()
        aa[8, 7] += (0.5**2*2*z*y*(y**2-z**2)).sum()
        aa[8, 8] += (0.5**2*4*y**2*z**2).sum()

        v[0] += m.flat[ipix].sum()
        v[1] += (m.flat[ipix] * x).sum()
        v[2] += (m.flat[ipix] * y).sum()
        v[3] += (m.flat[ipix] * z).sum()
        v[4] += (m.flat[ipix] * 0.5*(x**2-z**2)).sum()
        v[5] += (m.flat[ipix] * 0.5*2*x*y).sum()
        v[6] += (m.flat[ipix] * 0.5*2*x*z).sum()
        v[7] += (m.flat[ipix] * 0.5*(y**2-z**2)).sum()
        v[8] += (m.flat[ipix] * 0.5*2*y*z).sum()
        
    aa[0, 1] = aa[1, 0]
    aa[0, 2] = aa[2, 0]
    aa[0, 3] = aa[3, 0]
    aa[0, 4] = aa[4, 0]
    aa[0, 5] = aa[5, 0]
    aa[0, 6] = aa[6, 0]
    aa[0, 7] = aa[7, 0]
    aa[0, 8] = aa[8, 0]
    aa[1, 2] = aa[2, 1]
    aa[1, 3] = aa[3, 1]
    aa[1, 4] = aa[4, 1]
    aa[1, 5] = aa[5, 1]
    aa[1, 6] = aa[6, 1]
    aa[1, 7] = aa[7, 1]
    aa[1, 8] = aa[8, 1]
    aa[2, 3] = aa[3, 2]
    aa[2, 4] = aa[4, 2]
    aa[2, 5] = aa[5, 2]
    aa[2, 6] = aa[6, 2]
    aa[2, 7] = aa[7, 2]
    aa[2, 8] = aa[8, 2]
    aa[3, 4] = aa[4, 3]
    aa[3, 5] = aa[5, 3]
    aa[3, 6] = aa[6, 3]
    aa[3, 7] = aa[7, 3]
    aa[3, 8] = aa[8, 3]
    aa[4, 5] = aa[5, 4]
    aa[4, 6] = aa[6, 4]
    aa[4, 7] = aa[7, 4]
    aa[4, 8] = aa[8, 4]
    aa[5, 6] = aa[6, 5]
    aa[5, 7] = aa[7, 5]
    aa[5, 8] = aa[8, 5]
    aa[6, 7] = aa[7, 6]
    aa[6, 8] = aa[8, 6]
    aa[7, 8] = aa[8, 7]
    
    res = np.dot(np.linalg.inv(aa), v)
    mono = res[0]
    dipole = res[1:4]
    quad = np.zeros ((3,3), dtype=np.float64)
    quad [0,0] = res[4]
    quad [1,0] = res[5]
    quad [2,0] = res[6]
    quad [0,1] = res[5]
    quad [0,2] = res[6]
    quad [1,1] = res[7]
    quad [2,1] = res[8]
    quad [1,2] = res[8]
    quad [2,2] = -res[4]-res[7]
    #quad /= 2
    return mono, dipole, quad
