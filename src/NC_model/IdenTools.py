import numpy as np
from neuron.units import mV, ms

def MovingAverage(array):
    res = np.zeros_like(array)
    mean = 0
    for i in range(len(array)):
        mean = (mean*i + array[i])/(i+1)
        res[i] = mean
    return res
def LocalAverage(array, num_points):
    res = np.zeros_like(array)
    array = np.concatenate((array[-num_points+1:], array, array[:num_points-1]))
    local_mean = array[:num_points].mean()
    for i in range(len(res)):
        local_mean += (array[i + num_points] - array[i])/num_points
        res[i] = local_mean
    return res
def IsZero(arg, array):
    leftarg = arg - 1 if arg >= 1 else 0
    rightarg = arg + 1 if arg + 1 < len(array) else len(array) - 1
    passzero = np.sign(array[leftarg]) != np.sign(array[rightarg])
    norm_arr = np.abs(array)
    ismin = (norm_arr[arg] == np.min(norm_arr[leftarg:rightarg+1]))
    return ismin & (passzero or abs(array[arg]) < 1e-10)

def SummitAndTrough(ts,vs,SPIKE_DIFF=20):
    dvs = np.concatenate((np.nan, (vs[2:] - vs[:-2])/2, np.nan), axis=None)
    ddvs = np.concatenate((np.nan, (dvs[2:] - dvs[:-2])/2, np.nan), axis=None)
    inert = vs - LocalAverage(vs,500) <= SPIKE_DIFF
    extreme = [IsZero(i,dvs) for i in range(len(dvs))]
    up = (ddvs < 0*mV)
    summit_cond = ~inert & extreme & up
    trough_cond =  extreme & ~up
    return summit_cond, trough_cond
def SpikeTime(ts,summit_cond):
    return ts[summit_cond]
def FirstSpikeLatency(ts_spike, START_TIME=200):
    try: assert np.shape(ts_spike)[0] != 0
    except: return -1
    return ts_spike[0] - START_TIME
def InterSpikeIntervals(ts_spike):
    try: assert np.shape(ts_spike)[0] != 0
    except: return []
    return ts_spike[1:] - ts_spike[:-1]
def PostSpikeSilence(ts_spike, STOP_TIME=800):
    try: assert np.shape(ts_spike)[0] != 0
    except: return -1
    return STOP_TIME - ts_spike[-1]
def SlowWaveAmplitude(ts, vs, trough_cond, STOP_TIME=800):
    try: assert np.shape(trough_cond)[0] != 0
    except: return -1
    return (np.max(vs[trough_cond]) - vs[ts==STOP_TIME])[0]
def GetProp(ts, vs):
    s,t = SummitAndTrough(ts,vs)
    spts = SpikeTime(ts, s)
    fsl = FirstSpikeLatency(spts)
    isis = InterSpikeIntervals(spts)
    pss = PostSpikeSilence(spts)
    swa = SlowWaveAmplitude(ts,vs,t)
    return np.array([fsl, pss, swa, *isis, *np.zeros(1000-len(isis)-3)])

def HasDelay(delay, isis, DELAY_FACTOR=2):
    return True if delay > (isis[0] + isis[-1])*DELAY_FACTOR/2 else False
def HasTSWBorTSTUT(isis, swa, pss, PRE_FACTOR=2.5, POST_FACTOR=1.5, HIGH_FREQ=25, MIN_SWA=5):
    if np.shape(isis)[0] < 5: return False
    for i in [2,3,4]:
        if (isis[i] > isis[i-1]*PRE_FACTOR) & (isis[i] > isis[i+1]*POST_FACTOR) & \
            (np.mean(isis[i:]) > np.mean(isis[:i-1]) * PRE_FACTOR) & \
            np.all(1e3/isis[:i] >= HIGH_FREQ):
            return True
    if (pss > isis[-1]*PRE_FACTOR) & np.all(1e3/isis[:i] >= HIGH_FREQ) & (swa > MIN_SWA):
        return True
    else:
        return False
def HasTSWB(swa, MIN_SWA=5):
    if swa > MIN_SWA: return True
    else: return False
def RunSolverStatTests():
    return True
def HasPSTUT(isis, FACTOR=5):
    i = np.argmax(isis)
    if (isis[i]/isis[i-1] + isis[i]/isis[i+1] > FACTOR): return True
    else: return False
def HasSLN(pss, isis, FACTOR=2):
    if (pss > FACTOR*np.mean(isis)) & (pss > FACTOR*np.max(isis)):
        return True
    else: return False