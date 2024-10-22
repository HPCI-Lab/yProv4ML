
import geocoder
import pandas as pd 
from codecarbon import EmissionsTracker
from typing import Any, Callable, Tuple

CARBON_INTENSITY = pd.read_csv('./prov4ml/utils/carbon_intensity.csv').set_index("Energy Source").transpose()
ENERGY_MIX = pd.read_csv('./prov4ml/utils/energy_mix_percs.csv').set_index("Country code")

def carbon_tracked_function(f: Callable, *args, **kwargs) -> Tuple[Any, Any]:
    """
    Tracks carbon emissions for a given function call.
    
    Args:
        f (Callable): The function to be executed and carbon emissions tracked.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
    
    Returns:
        Tuple[Any, Any]: A tuple containing the result of the function call and the total emissions tracked.
    """
    TRACKER.start()
    result = f(*args, **kwargs)
    _ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return result, total_emissions

def _carbon_init() -> None:
    """Initializes the carbon emissions tracker."""
    global TRACKER
    TRACKER = EmissionsTracker(
        save_to_file=False,
        save_to_api=False,
        save_to_logger=False, 
        log_level="error",
    ) #carbon emission tracker, don't save anywhere, just get the emissions value to log with prov4ml
    TRACKER.start()

def stop_carbon_tracked_block() -> Any:
    """
    Stops the tracking of carbon emissions for a code block and returns the total emissions tracked.
    
    Returns:
        Any: The total emissions tracked.
    """
    _ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return total_emissions

def get_country():
    g = geocoder.ip('me')
    if g.country is not None:
        return g.country
    else:
        return None

def get_carbon_emission(energy): 
    current_country = get_country()

    if current_country is None:
        world_average_CI = 475 # gCO2.eq/KWh
        return world_average_CI * energy
    else:
        country_em = ENERGY_MIX[ENERGY_MIX["ISO2"] == current_country]
        country_em = country_em.drop(columns=["ISO2", "ISO3"], inplace=False)

        if len(country_em) == 0:
            world_average_CI = 475
            return world_average_CI * energy
        else:
            weighted_CI = 0
            for col in country_em.columns:
                weighted_CI += country_em[col].values * CARBON_INTENSITY[col].values

            weighted_CI = weighted_CI[0]
            return weighted_CI * energy # gCO2.eq

def get_ce_delta_compared_to_world_avg(energy):
    ce = get_carbon_emission(energy) # gCO2.eq

    world_average_CI = 475 # gCO2.eq/KWh
    wce = world_average_CI * energy # gCO2.eq

    if wce == 0:
        return 0
    # return percentage difference between country and world average
    return ((ce - wce) / wce) * 100

def calc_world_average_CI():
    world_average_CI = 0
    for col in CARBON_INTENSITY.columns:
        world_average_CI += ENERGY_MIX[col].values * CARBON_INTENSITY[col].values

    world_average_CI = world_average_CI[0]
    return world_average_CI


