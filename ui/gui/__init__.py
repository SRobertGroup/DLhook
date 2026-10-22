"""Mixins that make up seedling_measurement.Gui, one per responsibility. Gui inherits them all, so
every method stays reachable as Gui.<name> (tests and multi/validate_angles.py build Gui.__new__)."""
from ui.gui.crop_boxes import CropBoxMixin
from ui.gui.export import ExportMixin
from ui.gui.scale import ScaleMixin
from ui.gui.seedling_windows import SeedlingWindowsMixin

__all__ = ["CropBoxMixin", "ExportMixin", "ScaleMixin", "SeedlingWindowsMixin"]
