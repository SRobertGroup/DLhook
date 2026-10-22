"""The GUI module was renamed seedling_measurment -> seedling_measurement (audit L-5); the old name
is a shim that resolves to the same module object."""


def test_old_module_name_is_the_same_module():
    import seedling_measurement
    import seedling_measurment

    assert seedling_measurment is seedling_measurement
    assert seedling_measurment.Gui is seedling_measurement.Gui
