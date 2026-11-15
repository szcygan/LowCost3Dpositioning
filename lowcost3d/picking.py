"""Interactive marker picking and result display (matplotlib).

Replaces ``getpts`` / ``imshowpair`` from the MATLAB scripts. Compared with
``getpts`` the picker numbers every click, allows undo, enforces the expected
number of markers and leaves the toolbar zoom/pan usable for precise clicks.
"""

from __future__ import annotations

import numpy as np


class PickCancelled(RuntimeError):
    """The user closed the window or pressed Escape."""


def pick_points(image: np.ndarray, title: str, n_expected: int | None = None,
                reference: np.ndarray | None = None) -> np.ndarray:
    """Let the user click marker seeds; returns an (N, 2) array of (x, y).

    Controls: left click = add, right click / Backspace = undo last,
    Enter = finish, Escape = cancel. Clicks are ignored while the zoom or pan
    tool is active, so you can zoom in for precise placement.
    """
    import matplotlib.pyplot as plt

    if reference is not None:
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(15, 6.5),
                                      gridspec_kw={"width_ratios": [3, 1]})
        axr.imshow(reference)
        axr.set_title("Marker order")
        axr.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()
    pts: list[tuple[float, float]] = []
    artists: list = []
    state = {"done": False, "cancelled": False}

    def status() -> str:
        cnt = f"{len(pts)}/{n_expected}" if n_expected else str(len(pts))
        return (f"{title}\nclicked {cnt} | left click: add, right click/Backspace: undo, "
                "Enter: done, Esc: cancel")

    ax.set_title(status(), fontsize=10)

    def redraw():
        ax.set_title(status(), fontsize=10)
        fig.canvas.draw_idle()

    def undo():
        if pts:
            pts.pop()
            for a in artists.pop():
                a.remove()
            redraw()

    def on_click(event):
        if event.inaxes is not ax:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan active
        if event.button == 1:
            if n_expected and len(pts) >= n_expected:
                ax.set_title(status() + "\nAll markers clicked - press Enter "
                             "(or undo)", fontsize=10, color="tab:red")
                fig.canvas.draw_idle()
                return
            pts.append((event.xdata, event.ydata))
            dot = ax.plot(event.xdata, event.ydata, "+", color="lime", ms=14, mew=2)[0]
            txt = ax.text(event.xdata + 8, event.ydata - 8, str(len(pts)), color="yellow",
                          fontsize=12, fontweight="bold")
            artists.append((dot, txt))
            redraw()
        elif event.button == 3:
            undo()

    def on_key(event):
        if event.key in ("backspace", "delete"):
            undo()
        elif event.key == "enter":
            if n_expected and len(pts) != n_expected:
                ax.set_title(status() + f"\nNeed exactly {n_expected} markers",
                             fontsize=10, color="tab:red")
                fig.canvas.draw_idle()
                return
            state["done"] = True
            plt.close(fig)
        elif event.key == "escape":
            state["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if not state["done"]:
        raise PickCancelled("point picking cancelled")
    return np.array(pts, dtype=float)


def show_segmentation(views, suptitle: str = "", block: bool = True, save: str | None = None):
    """Overlay masks and centroids. ``views`` = [(title, image, MarkerSegmentation), ...]."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(views), figsize=(8 * len(views), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (title, image, seg) in zip(axes, views):
        rgb = np.dstack([image] * 3)
        mask = seg.combined_mask
        rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([1.0, 0.0, 1.0])
        ax.imshow(rgb)
        ax.plot(seg.centroids[:, 0], seg.centroids[:, 1], "o", mfc="lime", mec="k", ms=6)
        for k, (x, y) in enumerate(seg.centroids, 1):
            ax.text(x + 10, y - 10, str(k), color="yellow", fontsize=11, fontweight="bold")
        pad = 80
        x0, y0 = seg.centroids.min(axis=0) - pad
        x1, y1 = seg.centroids.max(axis=0) + pad
        ax.set_xlim(max(x0, 0), min(x1, image.shape[1]))
        ax.set_ylim(min(y1, image.shape[0]), max(y0, 0))
        ax.set_title(title)
        ax.set_axis_off()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    if block:
        plt.show()
    else:
        plt.close(fig)
    return fig
