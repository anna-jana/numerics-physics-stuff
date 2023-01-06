import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip

fname = "catalog.dat"

def download():
    # https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/bsc5p.html
    url = "https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz"
    req = urllib.request.urlopen(url)
    gzfile = gzip.open(req)

    field_ranges = [
        (  1,  4), (  5, 14), ( 15, 25), ( 26, 31), ( 32, 37), ( 38, 41),
        ( 42, 42), ( 43, 43), ( 44, 44), ( 45, 49), ( 50, 51), ( 52, 60),
        ( 61, 62), ( 63, 64), ( 65, 68), ( 69, 69), ( 70, 71), ( 72, 73),
        ( 74, 75), ( 76, 77), ( 78, 79), ( 80, 83), ( 84, 84), ( 85, 86),
        ( 87, 88), ( 89, 90), ( 91, 96), ( 97,102), (103,107), (108,108),
        (109,109), (110,114), (115,115), (116,120), (121,121), (122,126),
        (127,127), (128,147), (148,148), (149,154), (155,160), (161,161),
        (162,166), (167,170), (171,174), (175,176), (177,179), (180,180),
        (181,184), (185,190), (191,194), (195,196), (197,197),
     ]
    field_ranges = [(a - 1, b) for a, b in field_ranges]

    field_names = [
      "HR", "Name", "DM", "HD", "SAO", "FK5", "IRflag", "r_IRflag", "Multiple",
      "ADS", "ADScomp", "VarID", "RAh1900", "RAm1900", "RAs1900", "DE-1900",
      "DEd1900", "DEm1900", "DEs1900", "RAh", "RAm", "RAs", "DE-", "DEd",
      "DEm", "DEs", "GLON", "GLAT", "Vmag", "n_Vmag", "u_Vmag", "B-V", "u_B-V",
      "U-B", "u_U-B", "R-I", "n_R-I", "SpType", "n_SpType", "pmRA", "pmDE",
      "n_Parallax", "Parallax", "RadVel", "n_RadVel", "l_RotVel", "RotVel",
      "u_RotVel", "Dmag", "Sep", "MultID", "MultCnt", "NoteFlag",
    ]

    d = pd.read_fwf(gzfile, colspecs=field_ranges,
            header=None, names=field_names)

    d.to_csv(fname)

def load():
    return pd.read_csv(fname, index_col=0)

def plot(d):
    # calculate the equatorial coordinates of the stars in radians
    # RA: right ascention
    # Dec: Declination
    # epoch: 2000
    # these are RA units (like time) NOT arcminutes, arcseconds etc. !!!
    ra = (d.RAh + d.RAm / 60 + d.RAs / 60**2) * 2*np.pi / 24
    # these are degrees, arcminutes, arcseconds
    dec_sign = np.where(d["DE-"] == "+", +1, -1)
    dec = dec_sign * (d.DEd + d.DEm / 60 + d.DEs / 60**2) * 2*np.pi / 360

    # visual magnitude
    mag = d.Vmag

    # color
    #B_minus_V = d["B-V"]
    #U_minus_B = d["U-B"]
    #R_minus_I = d["R-I"]

    brightness = 10**(-mag)
    brightness = (
        (brightness - brightness.min()) /
        (brightness.max() - brightness.min())
    )
    brightness[~np.isfinite(brightness)] = 0.1

    nh = dec >= 0.0 # northern hemissphere

    fig = plt.figure(layout="constrained")
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    ax1.scatter(ra[nh], np.cos(dec[nh]), s=1e4*brightness[nh],
            alpha=np.minimum(1, 0.3 + 1 - brightness[nh]), c="red")
    ax1.scatter(ra[nh], np.cos(dec[nh]), s=0.01, c="red")
    ax1.set_title("norther hemisphere")
    ax2.scatter(ra[~nh], np.cos(dec[~nh]), s=1e4*brightness[~nh],
            alpha=np.minimum(1, 0.3 + 1 - brightness[~nh]), c="red")
    ax2.scatter(ra[~nh], np.cos(dec[~nh]), s=0.01, c="red")
    ax2.set_title("souther hemisphere")

if __name__ == "__main__":
    if not os.path.exists(fname):
        download()
    d = load()
    plot(d)
    plt.show()
