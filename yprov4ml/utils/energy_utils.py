
from typing import Any, Callable, Tuple
from codecarbon import EmissionsTracker

def carbon_tracked_function(f: Callable, *args, **kwargs) -> Tuple[Any, Any]:
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
    total_emissions = TRACKER._prepare_emissions_data()
    return total_emissions
