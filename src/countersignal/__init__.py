"""CounterSignal: performance monitoring for deployed clinical risk models.

Corrects for the interventions the model's own alerts trigger, and reports
uncertainty intervals, small sample verdicts and sequential drift alarms.
"""

from .monitor import AWAITING, DRIFT, INSUFFICIENT, STABLE, Monitor, MonitorConfig, MonitorResult

__all__ = ["AWAITING", "DRIFT", "INSUFFICIENT", "STABLE", "Monitor", "MonitorConfig", "MonitorResult"]
__version__ = "0.2.0"
