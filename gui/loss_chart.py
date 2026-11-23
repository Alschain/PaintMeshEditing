"""Lightweight live loss curve plotter drawn directly on a Tkinter Canvas."""

import tkinter as tk


SERIES_COLORS = [
    '#60a5fa',  # total / blue
    '#34d399',  # emerald
    '#f472b6',  # pink
    '#fbbf24',  # amber
    '#a78bfa',  # violet
    '#fb7185',  # rose
    '#2dd4bf',  # teal
]


class LossChart(tk.Canvas):
    def __init__(self, parent, height=130, bg='#18191e', **kwargs):
        super().__init__(parent, height=height, bg=bg, highlightthickness=1, highlightbackground='#333742', **kwargs)
        self.bg = bg
        self.iters = []
        self.total_losses = []
        self.sub_losses = {}
        self.bind('<Configure>', lambda _e: self.redraw())
        self.redraw()

    def clear(self):
        self.iters.clear()
        self.total_losses.clear()
        self.sub_losses.clear()
        self.redraw()

    def push(self, iteration, total_loss, losses_dict=None):
        self.iters.append(int(iteration))
        self.total_losses.append(float(total_loss))
        if losses_dict:
            for k, v in losses_dict.items():
                self.sub_losses.setdefault(k, []).append(float(v))
        # Keep recent 300 points so redraw stays snappy
        if len(self.iters) > 300:
            self.iters = self.iters[-300:]
            self.total_losses = self.total_losses[-300:]
            for k in list(self.sub_losses.keys()):
                self.sub_losses[k] = self.sub_losses[k][-300:]
        self.redraw()

    def redraw(self):
        self.delete('all')
        w = max(self.winfo_width(), 200)
        h = max(self.winfo_height(), 100)

        pad_l, pad_r, pad_t, pad_b = 42, 10, 18, 20
        pw = max(10, w - pad_l - pad_r)
        ph = max(10, h - pad_t - pad_b)

        # Grid lines
        for i in range(4):
            y = pad_t + ph * i / 3.0
            self.create_line(pad_l, y, w - pad_r, y, fill='#282b34', dash=(2, 3))

        self.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline='#3a3f4d')

        if len(self.total_losses) < 2:
            self.create_text(
                w / 2,
                h / 2,
                text="Loss curve will appear during optimization",
                fill='#6b7280',
                font=('Segoe UI', 8),
            )
            return

        vals = self.total_losses
        v_min = min(vals)
        v_max = max(vals)
        if abs(v_max - v_min) < 1e-9:
            v_max = v_min + 1.0

        # Axis labels
        self.create_text(pad_l - 4, pad_t, text=f"{v_max:.3g}", anchor='ne', fill='#9ca3af', font=('Consolas', 7))
        self.create_text(pad_l - 4, h - pad_b, text=f"{v_min:.3g}", anchor='se', fill='#9ca3af', font=('Consolas', 7))
        self.create_text(pad_l, h - 4, text=f"it {self.iters[0]}", anchor='sw', fill='#9ca3af', font=('Consolas', 7))
        self.create_text(w - pad_r, h - 4, text=f"it {self.iters[-1]}", anchor='se', fill='#9ca3af', font=('Consolas', 7))

        # Plot total loss curve
        coords = []
        n = len(vals)
        for idx, val in enumerate(vals):
            x = pad_l + (idx / max(1, n - 1)) * pw
            y = pad_t + (1.0 - (val - v_min) / (v_max - v_min)) * ph
            coords.extend([x, y])

        self.create_line(*coords, fill=SERIES_COLORS[0], width=2, smooth=False)

        # Header badge
        self.create_text(
            pad_l + 4,
            4,
            text=f"Total Loss: {vals[-1]:.5f}",
            anchor='nw',
            fill=SERIES_COLORS[0],
            font=('Segoe UI', 8, 'bold'),
        )
