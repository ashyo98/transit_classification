'''
Functions for use in Transit Classification aglorithms
'''

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    """
    A simple linear trend for detrending.
    ----------
    Parameters:
        x (numpy array): x-values
        a (float): slope
        b (float): intercept
    ----------
    Returns:
        y (numpy array): linear function of x
    """
    return a*x + b

def stitch(lc_collection): 
    """
    Linearly detrend each "quarter" (interval of observing) of the data, 
    and combine them together. We do this to correct for offsets between quarters 
    and linear systmatics within each quarter. These are instrumental effects.
    ----------
    Parameters:
        lc_collection (lightcurve collection): the collection of quarters of lightcurve
                data returned by the Lightkurve query.
    ----------
    Returns:
        tot_time (numpy array): stitched time values
        tot_flux (numpy array): stitched and detrended flux values
        tot_flux_err (numpy array): stitched flux error values
        tot_qual (numpy array): stitched data quality flag values
    """
    tot_time = np.zeros(0) # to store concatenated arrays
    tot_flux = np.zeros(0)
    tot_flux_err = np.zeros(0)
    tot_qual = np.zeros(0)
    for i in range(len(lc_collection)):
        lc = lc_collection[i]
        flux = lc.flux.value
        time = lc.time.value
        flux_err = lc.flux_err.value
        qual = lc.quality
        rel_flux_err = flux_err/flux
        nan_mask = np.invert(np.isnan(flux))
        
        # Fit and remove linear trend
        popt, pcov = curve_fit(linear_func, time[nan_mask], flux[nan_mask])
        linear_trend = linear_func(time, *popt) # evaluate over the whole interval
        norm = flux / linear_trend
        
        tot_time = np.hstack([tot_time, time])
        tot_flux = np.hstack([tot_flux, norm])
        tot_flux_err = np.hstack([tot_flux_err, rel_flux_err])
        tot_qual = np.hstack([tot_qual, qual])
        
    return tot_time, tot_flux, tot_flux_err, tot_qual

def collect_curves(n_curves, n_timesteps=1000, pct_transit=48.8):
    """
    Collect raw light curves into flux and time arrays (of shape [n_curves, n_timesteps]).
    Construct corresponding array (of shape [n_curves]) containing labels (1 = transit, 0 = no transit).
    NOTE: downloads take a long time, so I recomend saving the resulting arrays as csv files.
    ----------
    Parameters:
        n_curves: number of curves to download (will be approximate if percentages don't work out)
        n_timesteps: number of timesteps to interpolate to (at some point we could also try truncation)
        pct_transit: percent of returned dataset that contains a transt. Default is the overall perentage
                    of the 150,000 available Keplar curves that have transits. 
    ----------
    Returns:
        all_curves: numpy array of shape [n_curves, n_timesteps] containing light curve flux values
        all_times: numpy array of shape [n_curves, n_timesteps] containing light curve time values
        all_labels: 1-dimensional array containing correponding transit labels (1 = transit, 0 = no transit)
    """

    # Get IDs of non-transit curves
    data = pd.read_csv('Data/exoplanet_archive_KOIs.csv')
    all_nontransit_ids = data.loc[data['koi_disposition'] == 'FALSE POSITIVE']['kepid'].to_list()
    nontransit_ids = np.random.choice(all_nontransit_ids, size = int(n_curves*(1-pct_transit/100)))
    all_labels = np.zeros(len(nontransit_ids))

    # Get IDs of transit curves 
    all_transit_ids = data.loc[data['koi_disposition'] == 'CONFIRMED']['kepid'].to_list()
    transit_ids = np.random.choice(all_transit_ids, size = int(n_curves*(pct_transit/100)))
    all_labels = np.concatenate((all_labels, np.ones(len(transit_ids))))

    # Fill array with transit and non-transit curves
    all_curves = np.zeros((len(nontransit_ids)+len(transit_ids), n_timesteps))
    all_times = np.zeros((len(nontransit_ids)+len(transit_ids), n_timesteps))
    i = 0
    for star in nontransit_ids:
        star = np.int(star)
        # Download full light curve
        curve = lk.search_lightcurve(f'KIC{star}', author='Kepler', cadence='long').download_all()
        # "Stich" together quarters
        time, flux, flux_err, quality = stitch(curve)
        # Set poor quality data to NaN
        good = (quality == 0) * (flux_err > 0) * (np.isfinite(time)) * (np.isfinite(flux)) * (np.isfinite(flux_err))
        flux[np.invert(good)] = np.NaN
        # Add time and flux to arrays
        all_times[i] = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(time)), time)
        all_curves[i] = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(flux)), flux)
        i += 1
    for star in transit_ids:
        star = np.int(star)
        # Download full light curve
        curve = lk.search_lightcurve(f'KIC{star}', author='Kepler', cadence='long').download_all()
        # "Stich" together quarters
        time, flux, flux_err, quality = stitch(curve)
        # Set poor quality data to NaN
        good = (quality == 0) * (flux_err > 0) * (np.isfinite(time)) * (np.isfinite(flux)) * (np.isfinite(flux_err))
        flux[np.invert(good)] = np.NaN
        # Add time and flux to arrays
        all_times[i] = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(time)), time)
        all_curves[i] = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(flux)), flux)
        i += 1

    # Randomize arrays (same randomization)
    p = np.random.permutation(len(all_curves))
    all_curves = all_curves[p]
    all_times = all_times[p]
    all_labels = all_labels[p]

    return all_curves, all_times, all_labels