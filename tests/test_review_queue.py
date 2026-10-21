import numpy as np

from utils.review_queue import frame_review_reason, next_in_queue, queue_position, review_queue


def _frame(state="Closed", angle=150.0, **readout):
    return {"angle_dict": {1: angle}, "state_dict": {1: state},
            "landmarks": {1: {"junction": (0, 0), **readout}}}


def test_frame_review_reason_queues_spikes_and_flags_but_not_manual_frames():
    assert frame_review_reason(_frame(), 1) is None
    assert frame_review_reason(_frame(series_edit="rejected"), 1) == "spike"
    assert frame_review_reason(_frame(temporal_flag="collar+junction"), 1) == "collar+junction flag"
    assert frame_review_reason(_frame(series_edit="rejected", temporal_flag="collar"), 1) == "spike"
    assert frame_review_reason(_frame(series_edit="interpolated"), 1) is None
    # reviewed: applied, accepted or blanked
    assert frame_review_reason(_frame("Manual", series_edit="rejected"), 1) is None
    assert frame_review_reason(_frame("Manual", np.nan, temporal_flag="junction"), 1) is None
    # unreadable frames only on request
    gated = _frame("Unreadable", np.nan, gated=True)
    assert frame_review_reason(gated, 1) is None
    assert frame_review_reason(gated, 1, include_unreadable=True) == "unreadable"
    assert frame_review_reason(None, 1) is None
    assert frame_review_reason({"state_dict": {1: "Closed"}, "landmarks": {1: None}}, 1) is None


def test_review_queue_orders_by_seedling_then_frame_and_steps_across_seedlings():
    results = {
        1: [_frame(), {"angle_dict": {2: 1.0}, "state_dict": {2: "Closed"},
                       "landmarks": {2: {"series_edit": "rejected"}}}],
        0: [_frame(temporal_flag="collar"), None, _frame(), _frame(series_edit="rejected")],
    }
    queue = review_queue(results)                       # seed_id = crop_id + 1
    assert queue == [(0, 0, "collar flag"), (0, 3, "spike"), (1, 1, "spike")]
    assert next_in_queue(queue, 0, 0) == (0, 3, "spike")
    assert next_in_queue(queue, 0, 1) == (0, 3, "spike")          # from a frame not in the queue
    assert next_in_queue(queue, 0, 3) == (1, 1, "spike")          # into the next seedling
    assert next_in_queue(queue, 1, 1) is None
    assert next_in_queue(queue, 1, 0, step=-1) == (0, 3, "spike")
    assert next_in_queue(queue, 0, 0, step=-1) is None
    assert queue_position(queue, 0, 3) == 1 and queue_position(queue, 0, 2) is None
