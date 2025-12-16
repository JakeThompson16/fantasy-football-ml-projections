
from functools import lru_cache
import nflreadpy as nfl

# Caches and returns schedule data for seasons
@lru_cache(maxsize=None)
def load_schedule_data(*seasons):
    return nfl.load_schedules(list(seasons))