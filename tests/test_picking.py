"""Smoke test of the point picker with synthetic mouse/keyboard events."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402

from lowcost3d.picking import PickCancelled, pick_points  # noqa: E402


def _driver(actions):
    def fake_show(*_, **__):
        fig = plt.gcf()
        ax = fig.axes[0]
        fig.canvas.draw()
        for act in actions:
            if act[0] == "click":
                _, x, y, button = act
                px, py = ax.transData.transform((x, y))
                ev = MouseEvent("button_press_event", fig.canvas, px, py, button=button)
                fig.canvas.callbacks.process("button_press_event", ev)
            else:
                ev = KeyEvent("key_press_event", fig.canvas, act[1])
                fig.canvas.callbacks.process("key_press_event", ev)
    return fake_show


def test_click_undo_and_finish(monkeypatch):
    img = np.zeros((100, 200))
    actions = [("click", 10, 20, 1), ("click", 50, 60, 1), ("click", 99, 99, 3),  # undo
               ("click", 70, 30, 1), ("key", "enter")]
    monkeypatch.setattr(plt, "show", _driver(actions))
    pts = pick_points(img, "test", n_expected=2)
    np.testing.assert_allclose(pts, [[10, 20], [70, 30]], atol=1e-6)


def test_enter_requires_expected_count(monkeypatch):
    img = np.zeros((100, 200))
    monkeypatch.setattr(plt, "show", _driver([("click", 10, 20, 1), ("key", "enter")]))
    with pytest.raises(PickCancelled):  # Enter ignored, window never confirmed
        pick_points(img, "test", n_expected=3)


def test_escape_cancels(monkeypatch):
    monkeypatch.setattr(plt, "show", _driver([("key", "escape")]))
    with pytest.raises(PickCancelled):
        pick_points(np.zeros((10, 10)), "test")
