"""Interactive 2D painting, mask editing, zoom/pan, and live-preview canvas for PME."""

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class PaintCanvas(ttk.Frame):
    """Interactive image viewport supporting label painting, mask painting,
    eyedropper color sampling, undo/redo, zoom/pan, and multi-view overlays.
    """

    VIEW_MODES = (
        'Composite (Label + Mask)',
        'Rendered View',
        'Target Label Only',
        'Mask View',
        'Optimization Live',
        'Normal Map',
        'Depth Map',
    )

    def __init__(self, parent, on_color_picked=None, on_status_update=None, on_canvas_changed=None):
        super().__init__(parent)
        self.on_color_picked = on_color_picked
        self.on_status_update = on_status_update
        self.on_canvas_changed = on_canvas_changed

        # Image buffers (all BGR uint8 or grayscale uint8)
        self.rendered_bgr = None
        self.label_bgr = None
        self.mask_gray = None
        self.live_preview_bgr = None
        self.normal_vis_bgr = None
        self.depth_vis_bgr = None

        # Painting state
        self.tool = 'paint_label'  # paint_label, erase_label, eyedropper, paint_mask, erase_mask, pan
        self.brush_radius = 12
        self.brush_color_bgr = (0, 0, 255)  # Default red label in BGR
        self.auto_mark_mask = True
        self.view_mode = self.VIEW_MODES[0]
        self.overlay_alpha = 0.35

        # Viewport transform
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._tk_img = None
        self._img_item = None
        self._cursor_item = None

        # Interaction tracking
        self._is_drawing = False
        self._is_panning = False
        self._last_img_pt = None
        self._pan_start = None

        # Undo / Redo history: list of (label_bgr.copy(), mask_gray.copy())
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 25

        self.canvas = tk.Canvas(
            self,
            bg='#141519',
            highlightthickness=0,
            cursor='crosshair',
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind('<ButtonPress-1>', self._on_left_down)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_up)
        self.canvas.bind('<ButtonPress-2>', self._on_pan_down)
        self.canvas.bind('<B2-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_pan_up)
        self.canvas.bind('<ButtonPress-3>', self._on_pan_down)
        self.canvas.bind('<B3-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>', self._on_pan_up)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Leave>', self._on_mouse_leave)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-4>', lambda e: self._zoom_at(e.x, e.y, 1.15))
        self.canvas.bind('<Button-5>', lambda e: self._zoom_at(e.x, e.y, 1.0 / 1.15))
        self.canvas.bind('<Configure>', lambda _e: self.refresh())

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_rendered_image(self, img_bgr, reset_label=True, reset_mask=False, fit=True):
        self.rendered_bgr = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        if reset_label or self.label_bgr is None or self.label_bgr.shape[:2] != (h, w):
            self.label_bgr = img_bgr.copy()
        if reset_mask or self.mask_gray is None or self.mask_gray.shape[:2] != (h, w):
            self.mask_gray = np.zeros((h, w), dtype=np.uint8)
        self._undo_stack.clear()
        self._redo_stack.clear()
        if fit:
            self.after(20, self.fit_to_window)
        self.refresh()

    def set_label_image(self, label_bgr, push_history=True):
        if push_history and self.label_bgr is not None:
            self._push_undo()
        self.label_bgr = label_bgr.copy()
        h, w = label_bgr.shape[:2]
        if self.rendered_bgr is None:
            self.rendered_bgr = label_bgr.copy()
        if self.mask_gray is None or self.mask_gray.shape[:2] != (h, w):
            self.mask_gray = np.zeros((h, w), dtype=np.uint8)
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()

    def set_mask_image(self, mask_gray, push_history=True):
        if push_history and self.mask_gray is not None:
            self._push_undo()
        m = mask_gray[..., 0] if mask_gray.ndim == 3 else mask_gray.copy()
        if self.label_bgr is not None and m.shape[:2] != self.label_bgr.shape[:2]:
            h, w = self.label_bgr.shape[:2]
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        self.mask_gray = np.where(m > 127, 255, 0).astype(np.uint8)
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()

    def set_live_preview(self, preview_bgr):
        self.live_preview_bgr = preview_bgr.copy() if preview_bgr is not None else None
        if self.view_mode == 'Optimization Live':
            self.refresh()

    def set_normal_vis(self, normal_vis_bgr):
        self.normal_vis_bgr = normal_vis_bgr.copy() if normal_vis_bgr is not None else None
        if self.view_mode == 'Normal Map':
            self.refresh()

    def set_depth_vis(self, depth_vis_bgr):
        self.depth_vis_bgr = depth_vis_bgr.copy() if depth_vis_bgr is not None else None
        if self.view_mode == 'Depth Map':
            self.refresh()

    def set_view_mode(self, mode):
        if mode in self.VIEW_MODES:
            self.view_mode = mode
            self.fit_to_window()

    def set_tool(self, tool_name):
        self.tool = tool_name
        cursor_map = {
            'paint_label': 'crosshair',
            'erase_label': 'circle',
            'eyedropper': 'tcross',
            'paint_mask': 'crosshair',
            'erase_mask': 'circle',
            'pan': 'fleur',
        }
        self.canvas.configure(cursor=cursor_map.get(tool_name, 'crosshair'))

    def reset_label_to_render(self):
        if self.rendered_bgr is None:
            return
        self._push_undo()
        self.label_bgr = self.rendered_bgr.copy()
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()

    def clear_mask(self):
        if self.mask_gray is None:
            return
        self._push_undo()
        self.mask_gray[:] = 0
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _push_undo(self):
        if self.label_bgr is None or self.mask_gray is None:
            return
        self._undo_stack.append((self.label_bgr.copy(), self.mask_gray.copy()))
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return False
        self._redo_stack.append((self.label_bgr.copy(), self.mask_gray.copy()))
        self.label_bgr, self.mask_gray = self._undo_stack.pop()
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self._undo_stack.append((self.label_bgr.copy(), self.mask_gray.copy()))
        self.label_bgr, self.mask_gray = self._redo_stack.pop()
        self.refresh()
        if self.on_canvas_changed:
            self.on_canvas_changed()
        return True

    # ------------------------------------------------------------------
    # Active display image composition
    # ------------------------------------------------------------------

    def _get_display_bgr(self):
        if self.view_mode == 'Optimization Live' and self.live_preview_bgr is not None:
            return self.live_preview_bgr
        if self.view_mode == 'Normal Map' and self.normal_vis_bgr is not None:
            return self.normal_vis_bgr
        if self.view_mode == 'Depth Map' and self.depth_vis_bgr is not None:
            return self.depth_vis_bgr
        if self.view_mode == 'Rendered View' and self.rendered_bgr is not None:
            return self.rendered_bgr
        if self.view_mode == 'Target Label Only' and self.label_bgr is not None:
            return self.label_bgr
        if self.view_mode == 'Mask View' and self.mask_gray is not None:
            return cv2.cvtColor(self.mask_gray, cv2.COLOR_GRAY2BGR)

        # Default Composite mode: Rendered view + edited label + translucent green mask overlay
        base = self.label_bgr if self.label_bgr is not None else self.rendered_bgr
        if base is None:
            return None
        comp = base.copy()
        if self.mask_gray is not None:
            m = self.mask_gray > 127
            # If a pre-masked label.png was loaded (zero outside mask), composite over rendered view for clarity
            if (
                self.rendered_bgr is not None
                and self.rendered_bgr.shape == comp.shape
                and np.any(m)
                and np.any(~m)
                and int(comp[~m].max()) == 0
            ):
                comp[~m] = self.rendered_bgr[~m]
            if self.overlay_alpha > 0.01 and np.any(m):
                tint = np.zeros_like(comp)
                tint[:, :] = (80, 230, 120)  # emerald green tint in BGR
                alpha = float(np.clip(self.overlay_alpha, 0.0, 0.9))
                comp[m] = np.clip(
                    comp[m].astype(np.float32) * (1.0 - alpha) + tint[m].astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
        return comp

    # ------------------------------------------------------------------
    # Coordinate transforms & Zoom/Pan
    # ------------------------------------------------------------------

    def fit_to_window(self):
        disp = self._get_display_bgr()
        if disp is None:
            return
        cw = max(self.canvas.winfo_width(), 300)
        ch = max(self.canvas.winfo_height(), 300)
        ih, iw = disp.shape[:2]
        scale = min((cw - 24) / max(1, iw), (ch - 24) / max(1, ih))
        self.zoom = max(0.05, min(8.0, scale))
        self.pan_x = (cw - iw * self.zoom) * 0.5
        self.pan_y = (ch - ih * self.zoom) * 0.5
        self.refresh()

    def _canvas_to_image(self, cx, cy):
        ix = (cx - self.pan_x) / self.zoom
        iy = (cy - self.pan_y) / self.zoom
        return int(round(ix)), int(round(iy))

    def _zoom_at(self, cx, cy, factor):
        new_zoom = max(0.05, min(12.0, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        ix = (cx - self.pan_x) / self.zoom
        iy = (cy - self.pan_y) / self.zoom
        self.zoom = new_zoom
        self.pan_x = cx - ix * self.zoom
        self.pan_y = cy - iy * self.zoom
        self.refresh()

    def refresh(self):
        self.canvas.delete('img')
        disp = self._get_display_bgr()
        if disp is None:
            cw = max(self.canvas.winfo_width(), 300)
            ch = max(self.canvas.winfo_height(), 300)
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="Load a sample or mesh to start Paint Mesh Editing",
                fill='#6b7280',
                font=('Segoe UI', 12),
                tags='img',
            )
            return

        ih, iw = disp.shape[:2]
        dw = max(1, int(round(iw * self.zoom)))
        dh = max(1, int(round(ih * self.zoom)))

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        interp = Image.NEAREST if self.zoom >= 3.0 else Image.BILINEAR
        pil_img = Image.fromarray(rgb).resize((dw, dh), interp)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self._img_item = self.canvas.create_image(
            int(round(self.pan_x)),
            int(round(self.pan_y)),
            anchor='nw',
            image=self._tk_img,
            tags='img',
        )
        self.canvas.tag_lower('img')

    # ------------------------------------------------------------------
    # Painting operations
    # ------------------------------------------------------------------

    def _can_paint_current_view(self):
        return (
            self.label_bgr is not None
            and self.view_mode in ('Composite (Label + Mask)', 'Target Label Only', 'Mask View', 'Rendered View')
        )

    def _apply_stroke(self, p0, p1):
        if self.label_bgr is None or self.mask_gray is None:
            return
        r = max(1, int(self.brush_radius))
        thickness = max(1, r * 2)

        if self.tool == 'paint_label':
            col = tuple(int(c) for c in self.brush_color_bgr)
            cv2.line(self.label_bgr, p0, p1, col, thickness=thickness, lineType=cv2.LINE_AA)
            if self.auto_mark_mask:
                cv2.line(self.mask_gray, p0, p1, 255, thickness=thickness)
        elif self.tool == 'erase_label' and self.rendered_bgr is not None:
            h, w = self.label_bgr.shape[:2]
            stroke_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(stroke_mask, p0, p1, 255, thickness=thickness)
            sel = stroke_mask > 0
            self.label_bgr[sel] = self.rendered_bgr[sel]
            if self.auto_mark_mask:
                self.mask_gray[sel] = 0
        elif self.tool == 'paint_mask':
            cv2.line(self.mask_gray, p0, p1, 255, thickness=thickness)
        elif self.tool == 'erase_mask':
            cv2.line(self.mask_gray, p0, p1, 0, thickness=thickness)

    def _sample_color(self, ix, iy):
        src = self.label_bgr if self.label_bgr is not None else self.rendered_bgr
        if src is None:
            return
        h, w = src.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            b, g, r = [int(v) for v in src[iy, ix]]
            self.brush_color_bgr = (b, g, r)
            if self.on_color_picked:
                self.on_color_picked((b, g, r))

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _on_left_down(self, event):
        if self.tool == 'pan':
            self._on_pan_down(event)
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        if self.tool == 'eyedropper':
            self._sample_color(ix, iy)
            return

        if not self._can_paint_current_view():
            return

        self._push_undo()
        self._is_drawing = True
        self._last_img_pt = (ix, iy)
        self._apply_stroke((ix, iy), (ix, iy))
        self.refresh()

    def _on_left_drag(self, event):
        if self.tool == 'pan':
            self._on_pan_drag(event)
            return

        self._draw_brush_cursor(event.x, event.y)
        if not self._is_drawing:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        prev = self._last_img_pt or (ix, iy)
        self._apply_stroke(prev, (ix, iy))
        self._last_img_pt = (ix, iy)
        self.refresh()

    def _on_left_up(self, event):
        if self.tool == 'pan':
            self._on_pan_up(event)
            return
        if self._is_drawing:
            self._is_drawing = False
            self._last_img_pt = None
            if self.on_canvas_changed:
                self.on_canvas_changed()

    def _on_pan_down(self, event):
        self._is_panning = True
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _on_pan_drag(self, event):
        if not self._is_panning or self._pan_start is None:
            return
        sx, sy, px0, py0 = self._pan_start
        self.pan_x = px0 + (event.x - sx)
        self.pan_y = py0 + (event.y - sy)
        self.refresh()

    def _on_pan_up(self, _event):
        self._is_panning = False
        self._pan_start = None

    def _on_mouse_wheel(self, event):
        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        self._zoom_at(event.x, event.y, factor)

    def _on_mouse_move(self, event):
        self._draw_brush_cursor(event.x, event.y)
        if self.on_status_update:
            disp = self._get_display_bgr()
            if disp is not None:
                ix, iy = self._canvas_to_image(event.x, event.y)
                h, w = disp.shape[:2]
                if 0 <= ix < w and 0 <= iy < h:
                    b, g, r = [int(v) for v in disp[iy, ix]]
                    self.on_status_update(f"XY: ({ix}, {iy})  |  RGB: ({r}, {g}, {b})  |  {w}x{h}  |  Zoom: {int(self.zoom * 100)}%")
                else:
                    self.on_status_update(f"{w}x{h}  |  Zoom: {int(self.zoom * 100)}%")

    def _on_mouse_leave(self, _event):
        self.canvas.delete('cursor')

    def _draw_brush_cursor(self, cx, cy):
        self.canvas.delete('cursor')
        if self.tool in ('paint_label', 'erase_label', 'paint_mask', 'erase_mask') and self._can_paint_current_view():
            rad = max(2.0, self.brush_radius * self.zoom)
            outline = '#facc15' if 'mask' in self.tool else '#ffffff'
            self.canvas.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                outline=outline, width=1.5, tags='cursor',
            )
