"""Frames that need a human look after the analysis (the review queue).

A frame is queued when its model reading was rejected by the series filter (a spike,
`series_edit == "rejected"`) or flagged by the time-series checks (`temporal_flag`, collar and/or
junction). Frames the readability gate set aside (`gated`) are queued only with
`include_unreadable=True`: they are a large share of a plate and most of them are genuinely
unmeasurable. A Manual frame (applied landmarks, an accepted reading or a blanked frame) counts as
reviewed and is never queued.

The queue is derived from the results each time it is needed, never stored, so it follows every
correction and re-reconstruction (flags are re-derived on each one). Items are
(crop_id, frame_index, reason), sorted by seedling then frame; crop_id is 0-based."""

SPIKE = "spike"
UNREADABLE = "unreadable"


def frame_review_reason(result, seed_id, include_unreadable=False):
    """Why this frame needs a look ("spike", "collar flag", "junction flag", "collar+junction flag",
    "unreadable"), or None."""
    if not result or (result.get("state_dict") or {}).get(seed_id) == "Manual":
        return None
    readout = (result.get("landmarks") or {}).get(seed_id)
    if not readout:
        return None
    if readout.get("gated"):
        return UNREADABLE if include_unreadable else None
    if readout.get("series_edit") == "rejected":
        return SPIKE
    if readout.get("temporal_flag"):
        return f"{readout['temporal_flag']} flag"
    return None


def review_queue(frame_results_by_crop, include_unreadable=False):
    """[(crop_id, frame_index, reason)] over every analysed seedling (seed_id = crop_id + 1)."""
    queue = []
    for crop_id in sorted(frame_results_by_crop):
        for index, result in enumerate(frame_results_by_crop[crop_id] or []):
            reason = frame_review_reason(result, crop_id + 1, include_unreadable)
            if reason is not None:
                queue.append((crop_id, index, reason))
    return queue


def next_in_queue(queue, crop_id, frame_index, step=+1):
    """The first item after (step=+1) or before (step=-1) position (crop_id, frame_index), or None."""
    here = (crop_id, frame_index)
    if step > 0:
        return next((item for item in queue if item[:2] > here), None)
    return next((item for item in reversed(queue) if item[:2] < here), None)


def queue_position(queue, crop_id, frame_index):
    """0-based index of (crop_id, frame_index) in the queue, or None when it is not queued."""
    return next((k for k, item in enumerate(queue) if item[:2] == (crop_id, frame_index)), None)
