import queue


class ProgressReporter:
    """Thread-safe channel between a background worker and the Tk main loop.

    The worker thread calls report(**kwargs) with whatever it wants the GUI
    to know (e.g. value=, message=, done=). The GUI polls drain() from the
    main thread (via root.after) and applies updates to its widgets there,
    instead of touching Tk state directly from the worker thread.
    """

    def __init__(self):
        self._queue = queue.Queue()

    def report(self, **kwargs):
        self._queue.put(kwargs)

    def drain(self):
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events
