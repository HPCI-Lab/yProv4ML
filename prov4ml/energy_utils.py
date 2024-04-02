
from codecarbon import EmissionsTracker

def carbon_tracked_function(f, *args, **kwargs):
    TRACKER.start()
    result = f(*args, **kwargs)
    _ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return result, total_emissions

def _carbon_init():
    global TRACKER
    TRACKER = EmissionsTracker(save_to_file=False,save_to_api=False,save_to_logger=False) #carbon emission tracker, don't save anywhere, just get the emissions value to log with prov4ml
    TRACKER.start()

def stop_carbon_tracked_block():
    _ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return total_emissions
