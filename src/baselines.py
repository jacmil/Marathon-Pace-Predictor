import math

def riegel_predict(race_time: float, race_distance: float, target_distance: float, exponent: float | float=1.06) -> float:
    """Predict race time using Riegel formula

    Uses Riegel formula to predict race time.

    Args:
        race time: previous race time 
        race distance: distance of that previous race
        target distance: distance of race to be predicted
        exponent: default exponent used in riegel formula

    Returns:
        float predicted race time in minutes
    """

    return race_time * (target_distance / race_distance) ** 1.06

def vdot_from_race(race_time: float, race_distance: float, rounding: int = 2) -> float:
    """Get VDOT score for a runner

    Uses Jack Daniels VDOT formula to get VDOT score for a runner

    Args:
        race time: runner's previous race time
        race distance: distance of runner's previous race

    Returns:
        float rounded VDOT score for runner 
    """

    mmc = 1609.344

    S = race_distance * mmc / race_time

    vdot = (-4.60 + 0.182258 * S + 0.000104 * S**2) / (0.8 + 0.1894393 * math.exp(-0.012778 * race_time) + 0.2989558 * math.exp(-0.1932605 * race_time))
    
    return round(vdot, rounding)


def vdot_predict(vdot: float, distance: float, tol: float = 0.01) -> float:
    """Predicts race time from VDOT score

    Uses VDOT score to predict race time

    Args: 
        vdot: runner's vdot score
        distance: target distance to predict

    Returns:
        float of predicted time in minutes
    """

    # binary search (nice) for vdot 
    lo, hi = 1, 600
    while hi - lo > tol:
        mid = (lo + hi) / 2 
        if vdot_from_race(race_time=mid, race_distance=distance) > vdot:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

