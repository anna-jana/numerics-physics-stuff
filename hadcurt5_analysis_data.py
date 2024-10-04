import urllib.request, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

default_fname = "hadcrut5_anaylsis_data.hdf5"
default_key = "k"

def download(
    url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv",
    fname = default_fname,
    key = default_key,
):
    req = urllib.request.urlopen(url)
    dat = pd.read_csv(req, parse_dates=["Time"], index_col="Time")
    dat.rename(inplace=True, columns={
        "Anomaly (deg C)": "temp",
        "Lower confidence limit (2.5%)": "lower",
        "Upper confidence limit (97.5%)": "upper",
    })
    dat.to_hdf(fname, key)
    return dat

def load(fname = default_fname, key = default_key):
    return pd.read_hdf(fname, key)

def plot(dat):
    plt.figure(layout="constrained")
    plt.fill_between(dat.index, dat.lower, dat.upper, color="tab:blue", alpha=0.2, label="lowest/highest est.")
    plt.plot(dat.index, dat.temp, label="central est.")
    by_decade = dat.groupby((dat.index.year // 10) * 10).agg(np.mean)
    decades = pd.to_datetime(pd.DataFrame({"year": by_decade.index, "month":1, "day":1}))
    plt.plot(decades, by_decade.temp, color="blue", label="moving average over one decade")
    plt.xlabel("date")
    plt.ylabel("global temperature")
    plt.title("HadCRUD5 analysis")
    plt.legend()

if __name__ == "__main__":
    if not os.path.exists(default_fname):
        download()
    dat = load()
    plot(dat)
    plt.show()
