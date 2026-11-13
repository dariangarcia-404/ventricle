"""Minimal Tk click-pickers for seeds and circular ROIs.

Picks are always returned in original image pixel coordinates and the CLI saves
them as JSON, so any measurement can be re-run exactly with ``--rois``.
"""

from __future__ import annotations

import math

import numpy as np

from .imaging import to_image


def _tk():
    try:
        import tkinter as tk
        from tkinter import messagebox

        from PIL import ImageTk
    except ImportError as exc:  # pragma: no cover - depends on the Python build
        raise RuntimeError(
            "Interactive picking needs tkinter. Install a Python with Tk support, "
            "or write the ROI JSON by hand and pass --rois (see examples/)."
        ) from exc
    return tk, messagebox, ImageTk


def pick_circles(title: str, frame: np.ndarray, names: list[str], size: int = 560) -> dict:
    """For each name, click the ROI centre then a point on its edge."""
    tk, messagebox, ImageTk = _tk()
    image = to_image(frame)
    scale = size / max(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)))

    window = tk.Tk()
    window.title(title)
    prompt = tk.StringVar()
    tk.Label(window, textvariable=prompt, font=("Helvetica", 13)).pack(pady=4)
    canvas = tk.Canvas(window, width=image.width, height=image.height, background="black", highlightthickness=0)
    canvas.pack()
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    state = {"i": 0, "centre": None, "rois": {}}

    def update_prompt():
        if state["i"] >= len(names):
            prompt.set("Done. Press Save.")
        else:
            step = "centre" if state["centre"] is None else "edge"
            prompt.set(f"{names[state['i']]}: click the {step}")

    def click(event):
        if state["i"] >= len(names):
            return
        if state["centre"] is None:
            state["centre"] = (event.x, event.y)
            canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, outline="red", width=2)
        else:
            cx, cy = state["centre"]
            r = max(2.0, math.hypot(event.x - cx, event.y - cy))
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="red", width=2)
            canvas.create_text(cx + r + 4, cy, text=names[state["i"]], fill="red", anchor="w")
            state["rois"][names[state["i"]]] = {"x": cx / scale, "y": cy / scale, "radius": r / scale}
            state["i"] += 1
            state["centre"] = None
        update_prompt()

    def save():
        if state["i"] < len(names):
            messagebox.showwarning("Incomplete", "Place every ROI first.")
            return
        window.destroy()

    canvas.bind("<Button-1>", click)
    tk.Button(window, text="Save", command=save).pack(pady=4)
    update_prompt()
    window.mainloop()
    if state["i"] < len(names):
        raise RuntimeError("ROI selection was cancelled")
    return state["rois"]


def pick_seeds(frames: list[np.ndarray], thumb: int = 240, cols: int = 4) -> list[dict]:
    """Click inside the LV, then the RV, on every slice in order."""
    tk, messagebox, ImageTk = _tk()
    window = tk.Tk()
    window.title("ventricle: click LV then RV on each slice")
    prompt = tk.StringVar()
    tk.Label(window, textvariable=prompt, font=("Helvetica", 13)).pack(pady=4)
    rows = math.ceil(len(frames) / cols)
    cell = thumb + 10
    outer = tk.Frame(window)
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, width=cols * cell, height=min(800, rows * (cell + 18)), background="black",
                       scrollregion=(0, 0, cols * cell, rows * (cell + 18)))
    bar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")

    photos, tiles = [], []
    for s, frame in enumerate(frames):
        im = to_image(frame)
        ow = im.width
        im.thumbnail((thumb, thumb))
        x, y = (s % cols) * cell + 5, (s // cols) * (cell + 18) + 5
        photo = ImageTk.PhotoImage(im)
        photos.append(photo)
        canvas.create_image(x, y, image=photo, anchor="nw")
        canvas.create_text(x, y + im.height + 2, text=f"slice {s + 1}/{len(frames)}", fill="white", anchor="nw")
        tiles.append((x, y, im.width, im.height, ow / im.width))
    state = {"slice": 0, "label": "LV", "seeds": []}

    def update_prompt():
        if state["slice"] >= len(frames):
            prompt.set("All slices done. Press Save.")
        else:
            prompt.set(f"Slice {state['slice'] + 1}/{len(frames)}: click inside the {state['label']}")

    def click(event):
        if state["slice"] >= len(frames):
            return
        ex, ey = canvas.canvasx(event.x), canvas.canvasy(event.y)
        x, y, w, h, k = tiles[state["slice"]]
        if not (x <= ex < x + w and y <= ey < y + h):
            return
        color = "red" if state["label"] == "LV" else "cyan"
        canvas.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, outline=color, width=2)
        canvas.create_text(ex + 6, ey, text=state["label"], fill=color, anchor="w")
        state["seeds"].append({"slice": state["slice"], "label": state["label"], "x": (ex - x) * k, "y": (ey - y) * k})
        if state["label"] == "LV":
            state["label"] = "RV"
        else:
            state["label"], state["slice"] = "LV", state["slice"] + 1
        update_prompt()

    def save():
        if state["slice"] < len(frames):
            messagebox.showwarning("Incomplete", "Click LV then RV on every slice first.")
            return
        window.destroy()

    canvas.bind("<Button-1>", click)
    tk.Button(window, text="Save", command=save).pack(pady=4)
    update_prompt()
    window.mainloop()
    if state["slice"] < len(frames):
        raise RuntimeError("seed selection was cancelled")
    return state["seeds"]
