
import time

class ClockController:
    time_zone = 'US/Eastern'

def wait_until(condition_function, seconds_to_wait, msg="", increment=0.1):
    waitUntil = time.time() + seconds_to_wait
    while not condition_function():
        if time.time() > waitUntil:
            raise RuntimeError(msg)
        time.sleep(increment)
    