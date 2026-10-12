"""Readability gate for landmark-head angle readings -- pure python, no numpy needed at import.

The landmark ensemble reads an angle on every frame, including frames carrying no usable
information (the annotator "skipped" them). The old rule (junction peak < 0.2 -> missing) rejected
only 25% of such frames on 2,085 hand-annotated frames. This logistic gate does better:

    z = GATE_BIAS + w_peak*peak + w_tip*tip_peak + w_sd*member_theta_sd
    p_unreadable = 1 / (1 + exp(-z));  unreadable if p_unreadable >= GATE_THRESHOLD

Features: `peak` (max of the ensemble-averaged junction heatmap), `tip_peak` (max of the
ensemble-averaged cotyledon-tip heatmap; TIP_FILL when no member has a tip head) and
`member_theta_sd` (population std of the members' own thetas; MEMBER_SD_FILL = 90 when any member
failed to read -- in the data every such frame was a skipped one). A single model has no member
spread, so it is gated with sd = 0.0; the gate was not fitted on single-model readouts.

Fitted on the development sets (measured vs skipped frames) and checked on the independent
BerryBox v2 set: it rejects 68% of skipped frames there while keeping 99% of measured frames
(MAE of the kept measured frames 12.8 deg vs 13.9 deg without the gate). The constants below were
fitted by a scratch analysis on multi/results/gate_features.csv (fit stored in
multi/results/landmark_gate_fit.json; not read at runtime).

Learned "measurable" head (preferred). When the landmark models carry the image-level
"measurable" output (models/unet.py: cls_measurable, trained on the annotators' skip labels), the
readout has "measurable_p" and the gate uses it instead of the logistic above:
unreadable_p = 1 - measurable_p, unreadable if unreadable_p >= MEASURABLE_THRESHOLD. The logistic
stays the fallback for models without the head.

DLHOOK_LANDMARK_GATE=0 disables the gate (see gate_enabled).
"""
from __future__ import annotations

import math
import os

GATE_WEIGHTS = {"peak": -7.778380758685286, "tip_peak": -5.0715649494710995,
                "member_theta_sd": 0.025095593068359068}
GATE_BIAS = 6.519508170902701
GATE_THRESHOLD = 0.6546185430239393
TIP_FILL = 0.4419798254966736          # tip_peak when no member has a tip head
MEMBER_SD_FILL = 90.0                  # member_theta_sd when any member readout failed
ENV_VAR = "DLHOOK_LANDMARK_GATE"
# Gate on the learned head: unreadable when 1 - measurable_p reaches this. Set from the threshold that
# keeps 98% of the measured frames of the training runs' held-out seedlings
# (multi/results/measurable_fit.json, written by multi/validate_measurable.py --fit).
MEASURABLE_THRESHOLD = 0.5


def unreadable_probability(peak, tip_peak, member_theta_sd):
    """p(frame is unreadable); tip_peak None -> TIP_FILL."""
    tip = TIP_FILL if tip_peak is None else float(tip_peak)
    z = (GATE_BIAS + GATE_WEIGHTS["peak"] * float(peak) + GATE_WEIGHTS["tip_peak"] * tip
         + GATE_WEIGHTS["member_theta_sd"] * float(member_theta_sd))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def is_unreadable(peak, tip_peak, member_theta_sd, threshold=None):
    return unreadable_probability(peak, tip_peak, member_theta_sd) >= (
        GATE_THRESHOLD if threshold is None else threshold)


def gate_enabled(environ=None):
    """True unless DLHOOK_LANDMARK_GATE is 0/false/off/no."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV_VAR, "1")).strip().lower() not in ("0", "false", "off", "no")


def annotate_readout(readout, tip_hm=None, member_thetas=None):
    """Add the gate fields to a merged readout dict (in place; None stays None).

    `tip_hm`: the (merged) cotyledon-tip heatmap or None. `member_thetas`: one theta per ensemble
    member, None for a member whose readout failed; None for a single model (sd = 0.0, documented
    in the module docstring). Adds tip_peak, member_theta_sd, member_thetas, unreadable_p and
    unreadable (bool, ignores DLHOOK_LANDMARK_GATE: the switch is applied by the GUI)."""
    if readout is None:
        return None
    tip_peak = None if tip_hm is None else float(tip_hm.max())
    if member_thetas is None:
        sd = 0.0
    elif any(t is None for t in member_thetas):
        sd = MEMBER_SD_FILL
    else:
        sd = _pstd([float(t) for t in member_thetas])
    readout["tip_peak"] = tip_peak
    readout["member_theta_sd"] = sd
    readout["member_thetas"] = None if member_thetas is None else list(member_thetas)
    measurable = readout.get("measurable_p")
    if measurable is not None:          # the learned head decides when the models have it
        p = 1.0 - float(measurable)
        readout["unreadable_p"] = p
        readout["unreadable"] = bool(p >= MEASURABLE_THRESHOLD)
        readout["gate_source"] = "measurable"
        return readout
    p = unreadable_probability(readout.get("peak", 0.0), tip_peak, sd)
    readout["unreadable_p"] = p
    readout["unreadable"] = bool(p >= GATE_THRESHOLD)
    readout["gate_source"] = "logistic"
    return readout


def _pstd(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
