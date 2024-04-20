import numpy as np
from neuron.units import mV, ms
from scipy.stats import f
import pwlf

#CONSTANTS
NEGLEGTIBLE_AMOUNT = 1e-10
SPIKE_DIFF=20
HIGH_FREQ=25
RAPID=0.2
INERT=0.003
START_TIME=200
STOP_TIME=800
ALPHA=0.005 #significant level

#SOME USEFUL FUNCTIONS
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
def IsZero(array):
    leftarr = np.concatenate(([array[0]], array[:-1]))
    rightarr = np.concatenate((array[1:], [array[-1]]))
    norm_array = np.abs(array)
    ismin = (norm_array < np.abs(leftarr)) & (norm_array < np.abs(rightarr))
    passzero = np.sign(leftarr) != np.sign(rightarr)
    tiny = np.abs(array) < NEGLEGTIBLE_AMOUNT
    return ismin & (passzero | tiny)

#FIND VISUAL PROPERTIES
def SummitAndTrough(ts,vs,SPIKE_DIFF=SPIKE_DIFF):
    dvs = np.concatenate((np.nan, (vs[2:] - vs[:-2])/2, np.nan), axis=None)
    ddvs = np.concatenate((np.nan, (dvs[2:] - dvs[:-2])/2, np.nan), axis=None)
    inert = vs - LocalAverage(vs,500) <= SPIKE_DIFF
    extreme = IsZero(dvs)
    up = (ddvs < 0*mV)
    summit_cond = ~inert & extreme & up
    trough_cond =  extreme & ~up
    return summit_cond, trough_cond
def SpikeTime(ts,summit_cond):
    spts = ts[summit_cond]
    spts = spts[(spts > START_TIME) & (spts < STOP_TIME)]
    return spts
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

#IDENTIFY FIRING PATTERN ELEMENTS
def HasDelay(delay, isis, DELAY_FACTOR=2):
    return True if delay > (isis[0] + isis[-1])*DELAY_FACTOR/2 else False
def HasTSWBorTSTUT(isis, swa, pss, PRE_FACTOR=2.5, POST_FACTOR=1.5, HIGH_FREQ=25, MIN_SWA=5):
    for i in [2,3,4]:
        try:
            if (isis[i] > isis[i-1]*PRE_FACTOR) & (isis[i] > isis[i+1]*POST_FACTOR) & \
            (np.mean(isis[i:]) > np.mean(isis[:i-1]) * PRE_FACTOR) & \
            np.all(1e3/isis[:i] >= HIGH_FREQ):
                isis = isis[i:]
                return True, isis
            elif (pss > isis[-1]*PRE_FACTOR) & np.all(1e3/isis[:i] >= HIGH_FREQ) & (swa > MIN_SWA):
                return True, isis
        except:
            if (pss > isis[-1]*PRE_FACTOR) & np.all(1e3/isis[:] >= HIGH_FREQ) & (swa > MIN_SWA):
                return True, isis
    return False, isis
def HasTSWB(swa, MIN_SWA=5):
    if swa > MIN_SWA: return True
    else: return False
def RunSolverStatTests(fsl, isis, START_TIME=START_TIME, ALPHA=ALPHA, RAPID=RAPID, INERT=INERT):
    try: assert len(isis) > 4
    except: return '' #Not enough ISIs for Statistical Test

    spts = START_TIME + fsl + np.concatenate(([0], np.cumsum(isis)))
    fs = 1e3/isis
    pw = pwlf.PiecewiseLinFit(spts[:-1], fs)

    pw.fit(1)
    a, b = *pw.slopes, *pw.intercepts
    rss1 = pw.ssr if pw.ssr > 0 else NEGLEGTIBLE_AMOUNT

    pw.fit(2)
    a1, a2, b1, b2 = *pw.slopes, *pw.intercepts
    rss2 = pw.ssr if pw.ssr > 0 else NEGLEGTIBLE_AMOUNT

    df1 = len(isis) - 2
    df2 = len(isis) - 4
    f_score = (rss1 - rss2)/(df1 - df2)/(rss2/df2)
    p_value = f.cdf(f_score, df1, df2)
    pattern = ['RASP.', 'ASP.', 'NASP']

    if p_value < ALPHA:
        pat = pattern[np.argwhere(np.abs(a) >= (RAPID, INERT, 0))[0,0]]
        return "No Adaptation" if pat == 'NASP' else pat
    elif (1 - p_value) < ALPHA:
        pat1 = pattern[np.argwhere(np.abs(a1) >= (RAPID, INERT, 0))[0,0]]
        pat2 = pattern[np.argwhere(np.abs(a2) >= (RAPID, INERT, 0))[0,0]]
        return "No Adaptation" if pat1 == 'NASP' else pat1 + pat2
    else:
        return "No Adaptation"
def HasPSTUT(isis, FACTOR=5):
    try: assert len(isis) > 1
    except: return False
    i = np.argmax(isis)
    try: assert i < len(isis) - 1
    except: return False
    if (isis[i]/isis[i-1] + isis[i]/isis[i+1] > FACTOR): return True
    else: return False
def HasPSWB(isis, swa, FACTOR=5, MIN_SWA=5):
    try: assert len(isis) > 1
    except: return False
    i = np.argmax(isis)
    try: assert i < len(isis) - 1
    except: return False
    if (isis[i]/isis[i-1] + isis[i]/isis[i+1] > FACTOR) & (swa > MIN_SWA): return True
    else: return False
def HasFastTransient(isis, RAPID=RAPID):
    fs = 1e3/isis
    is_rapid = np.abs((fs[1:] - fs[:-1])/isis[:-1]) >= RAPID
    if np.any(is_rapid):
        final = np.argwhere(is_rapid).flatten()[-1]
        isis = isis[np.arange(len(isis)) > final]
        return True, isis
    else: return False, isis
def HasSLN(pss, isis, FACTOR=2):
    try: assert len(isis) > 0
    except: return False
    if (pss > FACTOR*np.mean(isis)) & (pss > FACTOR*np.max(isis)): return True
    else: return False

#FOR PIECEWISE LINEAR REGRESSION
def ResidualSumofSquares(ys, ys_true):
    return np.sum(np.square(ys - ys_true))
def TwoPieceLinear(x, a1, a2, b1, b2):
    x_cross = (b2-b1)/(a1-a2) if not ((a1 == a2) | ((a2 == b2) & (a2 == 0))) else np.inf
    return np.piecewise(x, [x<x_cross], [lambda x: a1*x + b1, lambda x: a2*x + b2])
def One(param, spts, fs):
    a1 = b2 = a2 = 0
    b1 = param[:1]
    ys = np.vectorize(TwoPieceLinear, excluded=[1,2,3,4])(spts[:-1], a1, a2, b1, b2)
    return ResidualSumofSquares(ys, fs)
def Two(params, spts, fs):
    b2 = a2 = 0
    b1, a1 = params[:2]
    ys = np.vectorize(TwoPieceLinear, excluded=[1,2,3,4])(spts[:-1], a1, a2, b1, b2)
    return ResidualSumofSquares(ys, fs)
def Three(params, spts, fs):
    a2 = 0
    b1, a1, b2 = params[:3]
    ys = TwoPieceLinear(spts[:-1], a1, a2, b1, b2)
    return ResidualSumofSquares(ys, fs)
def Four(params, spts, fs):
    b1, a1, b2, a2 = params[:4]
    ys = TwoPieceLinear(spts[:-1], a1, a2, b1, b2)
    return ResidualSumofSquares(ys, fs)

#FOR FIRING PATTERN IDENTIFICATION
def FirstStage(pattern, fsl, isis):
    if HasDelay(fsl, isis):
        pattern += "D."
    return pattern, isis
def SecondStage(pattern, swa, pss, isis):
    has, isis = HasTSWBorTSTUT(isis, swa, pss)
    if has:
        if HasTSWB(swa):
            pattern += "TSWB."
        else:
            pattern += "TSTUT."
    return pattern, isis
def ThirdStage(pattern, fsl, pss, swa, isis):
    res = RunSolverStatTests(fsl,isis)
    if res == "No Adaptation":
        if HasPSWB(isis,swa):
            pattern += "PSWB"
        elif HasPSTUT(isis):
            pattern += "PSTUT"
        else:
            has, isis = HasFastTransient(isis, RAPID)
            if has:
                pattern += "RASP."
                ThirdStage(pattern, fsl, pss, swa, isis)
            else:
                pattern += "NASP"
    else:
        pattern += res
        if (res != "ASP.NASP") & HasSLN(pss,isis):
            pattern += "SLN"
    return pattern, isis

def IdentifyPattern(fsl,swa,pss,isis):
    pattern = ''
    isis = isis[isis>0]
    try: assert len(isis) > 0
    except:
        if fsl > 0:
            pattern += 'D.'
        if pss > 0:
            pattern += 'SLN'
        return pattern
    
    pattern, isis = FirstStage(pattern, fsl, isis)
    pattern, isis = SecondStage(pattern, swa, pss, isis)
    pattern, isis = ThirdStage(pattern, fsl, pss, swa, isis)

    return pattern

#Firing pattern families
fpfamilies = {
    'regular' : ['D.SLN','NASP', 'ASP.','SLN'],
    'burst': ['TSWB.SLN', 'PSWB'],
    'inert': ['']
}