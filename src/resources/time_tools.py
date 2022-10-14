
import time
import arrow


class ClockController:
    time_zone = "US/Eastern"
    _simulated_time: arrow.Arrow = arrow.utcnow()

    @staticmethod
    def set_utcnow(simulated_time: arrow.Arrow):
        ClockController._simulated_time = simulated_time

    @staticmethod
    def increment_utcnow(increment_seconds: int):
        if ClockController._simulated_time is None:
            return
        ClockController._simulated_time = ClockController._simulated_time.shift(
            seconds=increment_seconds)

    @staticmethod
    def utcnow() -> arrow.Arrow:
        if ClockController._simulated_time is None:
            return arrow.utcnow()
        return ClockController._simulated_time


def wait_until(condition_function, seconds_to_wait: int, msg: str = "", increment: float = 0.001):
    waitUntil = time.time() + seconds_to_wait
    while not condition_function():
        if time.time() > waitUntil:
            raise RuntimeError(msg)
        time.sleep(increment)
