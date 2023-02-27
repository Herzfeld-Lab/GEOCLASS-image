import numpy as np

# TODO: make vertical window size param (40m?)
# TODO: document that vlen = lag and why

def generate_histo(data, hlen, vlen, avg_ht):
    # approach:
    #   filter data further by avg_ht +- 20m
    #   force vertical edges to always be same len: 40*.2 (or something similar)
    horizontal_edges = np.arange(np.nanmin(data[:, 4]), np.nanmax(data[:, 4]) + hlen, hlen)
    vertical_edges = np.arange(np.nanmin(data[:, 3]), np.nanmax(data[:, 3]) + 2 * vlen, vlen)# The times 2 is to avoid edge effects
    H = np.histogram2d(data[:, 4], data[:, 3], bins=(horizontal_edges, vertical_edges))[0]
    return H


def make_histo_features(ge, wp, windows, lag, vbin_size):

    # return: [segment_lat, segment_lon, nres along track histos...]

    wp_data = np.loadtxt(wp)
    ge_data = np.loadtxt(ge)
    distance = ge_data[:,3]
    elevation = ge_data[:,2]

    histos = []
    for w in range(len(windows)):
        start, end = windows[w,0], windows[w,1]
        window_bool = np.logical_and(distance>=start,distance<end)

        avg_ht = np.mean(elevation[window_bool])

        histo = generate_histo(wp_data[window_bool], hlen=lag, vlen=vbin_size)
        histos.append(histo)

