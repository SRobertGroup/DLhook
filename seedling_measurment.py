"""Old, misspelt name of seedling_measurement.py, kept so existing `import seedling_measurment` keeps
working. Both names are the SAME module object, so a monkeypatch through either name takes effect."""
import sys

import seedling_measurement as _module

sys.modules[__name__] = _module
