"""Interactive Studio GUI for Paint Mesh Editing (PME)."""

import os
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np

from core.ai_editor import AIImageEditor, compute_edit_mask
from gui.canvas import PaintCanvas
from gui.loss_chart import LossChart
from gui.viewport3d import Viewport3D
from gui.preview_renderer import (
    check_gpu_backend,
    load_depth_map_file,
    load_normal_map_file,
    render_mesh_view,
)
from util.vis_util import vis_depth_map, vis_normal_map


# Dark Studio Theme Palette
BG_DARK = '#18191e'
BG_PANEL = '#22242c'
BG_INPUT = '#2b2e38'
FG_TEXT = '#e5e7eb'
FG_MUTED = '#9ca3af'
ACCENT = '#3b82f6'
ACCENT_HOVER = '#2563eb'
SUCCESS = '#10b981'
DANGER = '#ef4444'
BORDER = '#343846'

LOSS_PRESETS = {
    'Default PME': {
        'img_loss': 100.0, 'img_l2_loss': 0.0, 'ssim_loss': 0.0, 'vgg_loss': 0.0,
        'normal_loss': 0.1, 'depth_loss': 0.0, 'silhouette_loss': 0.0,
        'nc_loss': 10.0, 'edge_loss': 100.0, 'area_loss': 100000.0,
        'lap_loss': 0.0, 'diff_lap_loss': 0.0, 'disp_loss': 0.0,
    },
    'Detail-Preserving': {
        'img_loss': 100.0, 'img_l2_loss': 20.0, 'ssim_loss': 15.0, 'vgg_loss': 0.0,
        'normal_loss': 5.0, 'depth_loss': 0.0, 'silhouette_loss': 0.0,
        'nc_loss': 15.0, 'edge_loss': 150.0, 'area_loss': 80000.0,
        'lap_loss': 0.0, 'diff_lap_loss': 50.0, 'disp_loss': 2.0,
    },
    'Multi-Modal (Normal+Depth+Sil)': {
        'img_loss': 80.0, 'img_l2_loss': 0.0, 'ssim_loss': 10.0, 'vgg_loss': 0.0,
        'normal_loss': 20.0, 'depth_loss': 15.0, 'silhouette_loss': 40.0,
        'nc_loss': 10.0, 'edge_loss': 100.0, 'area_loss': 50000.0,
        'lap_loss': 0.0, 'diff_lap_loss': 25.0, 'disp_loss': 1.0,
    },
    'Perceptual (VGG+SSIM)': {
        'img_loss': 60.0, 'img_l2_loss': 0.0, 'ssim_loss': 25.0, 'vgg_loss': 5.0,
        'normal_loss': 1.0, 'depth_loss': 0.0, 'silhouette_loss': 0.0,
        'nc_loss': 12.0, 'edge_loss': 120.0, 'area_loss': 80000.0,
        'lap_loss': 0.0, 'diff_lap_loss': 20.0, 'disp_loss': 1.0,
    },
}


class ScrollableFrame(ttk.Frame):
    """Vertical scrollable container for dense inspector tabs."""

    def __init__(self, parent, bg=BG_PANEL):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style='Panel.TFrame')

        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind('<Configure>', self._on_inner_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind('<Enter>', self._bind_wheel)
        self.canvas.bind('<Leave>', self._unbind_wheel)

    def _on_inner_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _bind_wheel(self, _e):
        self.canvas.bind_all('<MouseWheel>', self._on_wheel)

    def _unbind_wheel(self, _e):
        self.canvas.unbind_all('<MouseWheel>')

    def _on_wheel(self, event):
        # Only scroll if mouse is over the inspector panel (not main PaintCanvas)
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is not None and str(widget).startswith(str(self)):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')


class PaintMeshEditingApp(tk.Tk):
    def __init__(self, initial_sample=None):
        super().__init__()
        self.title("Paint Mesh Editing (PME) Studio")
        self.geometry("1440x880")
        self.minsize(1120, 700)
        self.configure(bg=BG_DARK)

        self.gpu_ok, self.gpu_status = check_gpu_backend()
        self.ai_editor = AIImageEditor()

        # Scene runtime state
        self.intrinsic = None
        self.extrinsic = None
        self.base_extrinsic = None
        self.last_render_fg_mask = None
        self.normal_target = None
        self.depth_target = None
        self.silhouette_target = None
        self.current_mesh = None  # PyTorch Mesh after optimization/render

        # Multi-view simultaneous optimization state
        self.multiview_list = []
        self._active_view_idx = None

        # Optimization thread control
        self._opt_thread = None
        self._stop_event = threading.Event()
        self._ai_busy = False

        self._setup_styles()
        self._init_variables()
        self._build_ui()
        self._bind_shortcuts()

        self.log(f"Environment: {self.gpu_status}")
        if initial_sample in ('cow', 'woman'):
            self.after(100, lambda: self.load_sample_preset(initial_sample))

    # ------------------------------------------------------------------
    # Styling & Tk Variables
    # ------------------------------------------------------------------

    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure('.', background=BG_PANEL, foreground=FG_TEXT, font=('Segoe UI', 9))
        style.configure('TFrame', background=BG_DARK)
        style.configure('Panel.TFrame', background=BG_PANEL)
        style.configure('Toolbar.TFrame', background='#1c1e26')
        style.configure('TLabel', background=BG_PANEL, foreground=FG_TEXT)
        style.configure('Muted.TLabel', background=BG_PANEL, foreground=FG_MUTED, font=('Segoe UI', 8))
        style.configure('Section.TLabel', background=BG_PANEL, foreground='#93c5fd', font=('Segoe UI', 9, 'bold'))
        style.configure('Status.TLabel', background='#14151a', foreground=FG_MUTED, font=('Consolas', 8))

        style.configure(
            'TButton',
            background='#313543',
            foreground=FG_TEXT,
            borderwidth=0,
            focusthickness=0,
            padding=(8, 4),
        )
        style.map('TButton', background=[('active', '#3f4558'), ('disabled', '#262933')])

        style.configure(
            'Accent.TButton',
            background=ACCENT,
            foreground='#ffffff',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 5),
        )
        style.map('Accent.TButton', background=[('active', ACCENT_HOVER), ('disabled', '#283655')])

        style.configure(
            'Success.TButton',
            background=SUCCESS,
            foreground='#06281e',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 5),
        )
        style.map('Success.TButton', background=[('active', '#059669'), ('disabled', '#1f3b32')])

        style.configure(
            'Danger.TButton',
            background=DANGER,
            foreground='#ffffff',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 5),
        )
        style.map('Danger.TButton', background=[('active', '#dc2626'), ('disabled', '#4b2323')])

        style.configure(
            'ToolActive.TButton',
            background=ACCENT,
            foreground='#ffffff',
            font=('Segoe UI', 8, 'bold'),
            padding=(4, 6),
        )
        style.configure(
            'Tool.TButton',
            background='#282b36',
            foreground=FG_TEXT,
            font=('Segoe UI', 8),
            padding=(4, 6),
        )

        style.configure('TEntry', fieldbackground=BG_INPUT, foreground=FG_TEXT, insertcolor=FG_TEXT, bordercolor=BORDER)
        style.configure('TCombobox', fieldbackground=BG_INPUT, background='#313543', foreground=FG_TEXT, arrowcolor=FG_TEXT)
        style.map('TCombobox', fieldbackground=[('readonly', BG_INPUT)], selectbackground=[('readonly', ACCENT)])
        style.configure('TCheckbutton', background=BG_PANEL, foreground=FG_TEXT)
        style.configure('TLabelframe', background=BG_PANEL, foreground='#93c5fd', bordercolor=BORDER)
        style.configure('TLabelframe.Label', background=BG_PANEL, foreground='#93c5fd', font=('Segoe UI', 9, 'bold'))
        style.configure('TNotebook', background=BG_DARK, borderwidth=0)
        style.configure('TNotebook.Tab', background='#262934', foreground=FG_MUTED, padding=(8, 6), font=('Segoe UI', 9))
        style.map('TNotebook.Tab', background=[('selected', BG_PANEL)], foreground=[('selected', '#ffffff')])
        style.configure('Horizontal.TProgressbar', troughcolor=BG_INPUT, background=ACCENT, bordercolor=BORDER)

    def _init_variables(self):
        # Scene & Data
        self.var_mesh_path = tk.StringVar()
        self.var_tex_path = tk.StringVar()
        self.var_intr_path = tk.StringVar()
        self.var_extr_path = tk.StringVar()
        self.var_label_path = tk.StringVar()
        self.var_mask_path = tk.StringVar()
        self.var_normal_path = tk.StringVar()
        self.var_depth_path = tk.StringVar()
        self.var_sil_path = tk.StringVar()

        self.var_render_h = tk.IntVar(value=1024)
        self.var_render_w = tk.IntVar(value=1024)
        self.var_device = tk.StringVar(value='cuda' if self.gpu_ok else 'cpu')
        self.var_rastype = tk.StringVar(value='opengl')
        self.var_use_multiview = tk.BooleanVar(value=True)
        self.var_view_weight = tk.DoubleVar(value=1.0)

        # Viewport & Paint
        self.var_view_mode = tk.StringVar(value=PaintCanvas.VIEW_MODES[0])
        self.var_overlay_alpha = tk.DoubleVar(value=0.35)
        self.var_brush_radius = tk.IntVar(value=12)
        self.var_auto_mark_mask = tk.BooleanVar(value=True)
        self.var_color_tol = tk.IntVar(value=45)
        self.var_diff_thresh = tk.IntVar(value=15)
        self.var_dilate_px = tk.IntVar(value=5)

        # AI 2D Editor
        self.var_ai_backend = tk.StringVar(value='local_refine')
        self.var_ai_prompt = tk.StringVar(value='smooth')
        self.var_ai_neg_prompt = tk.StringVar(value='blurry, distorted, low quality')
        self.var_ai_strength = tk.DoubleVar(value=0.65)
        self.var_ai_cfg = tk.DoubleVar(value=7.5)
        self.var_ai_steps = tk.IntVar(value=25)
        self.var_ai_seed = tk.IntVar(value=-1)
        self.var_ai_endpoint = tk.StringVar(value='http://127.0.0.1:7860')
        self.var_ai_model_id = tk.StringVar(value='runwayml/stable-diffusion-inpainting')
        self.var_ai_api_key = tk.StringVar(value='')

        # Optimization & Adaptive Loss
        self.var_iters = tk.IntVar(value=500)
        self.var_lr = tk.DoubleVar(value=3e-4)
        self.var_lr_decay = tk.DoubleVar(value=0.0002)
        self.var_log_interval = tk.IntVar(value=10)
        self.var_save_interval = tk.IntVar(value=50)
        self.var_save_mesh_interval = tk.IntVar(value=100)
        self.var_out_dir = tk.StringVar(value='./output')
        self.var_use_blur = tk.BooleanVar(value=False)
        self.var_blur_k = tk.IntVar(value=9)

        self.var_adaptive_mode = tk.StringVar(value='ema_scale')
        self.var_adapt_interval = tk.IntVar(value=10)
        self.var_adapt_ema = tk.DoubleVar(value=0.9)
        self.var_adapt_min = tk.DoubleVar(value=0.1)
        self.var_adapt_max = tk.DoubleVar(value=10.0)

        self.var_loss_preset = tk.StringVar(value='Default PME')
        self.loss_vars = {}
        for k, v in LOSS_PRESETS['Default PME'].items():
            self.loss_vars[k] = tk.DoubleVar(value=v)

        # Status
        self.var_status = tk.StringVar(value="Ready")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_progress_text = tk.StringVar(value="Idle")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # 1. Top Header / Action Bar
        topbar = ttk.Frame(self, style='Toolbar.TFrame', padding=(10, 6))
        topbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(
            topbar,
            text="PME Studio",
            background='#1c1e26',
            foreground='#ffffff',
            font=('Segoe UI', 11, 'bold'),
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(topbar, text="Cow Sample", command=lambda: self.load_sample_preset('cow')).pack(side=tk.LEFT, padx=2)
        ttk.Button(topbar, text="Woman Sample", command=lambda: self.load_sample_preset('woman')).pack(side=tk.LEFT, padx=2)
        ttk.Separator(topbar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Button(topbar, text="Render View", style='Accent.TButton', command=self.render_current_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(topbar, text="Auto Mask", command=self.generate_auto_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(topbar, text="Save Label & Mask", command=self.save_label_and_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(topbar, text="Export Mesh (.obj)", command=self.export_deformed_mesh).pack(side=tk.LEFT, padx=2)

        badge_color = '#10b981' if self.gpu_ok else '#f59e0b'
        tk.Label(
            topbar,
            text=f"● {self.gpu_status}",
            bg='#1c1e26',
            fg=badge_color,
            font=('Segoe UI', 8, 'bold'),
        ).pack(side=tk.RIGHT, padx=6)

        # 2. Main Body: Left Tool Strip + Center Canvas + Right Inspector
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Tool Strip
        self._build_tool_strip(body)

        # Center Viewport Column (2D Paint & Optimize + 3D Interactive Viewport)
        center = ttk.Frame(body)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 4), pady=4)

        self.center_tabs = ttk.Notebook(center)
        self.center_tabs.pack(fill=tk.BOTH, expand=True)

        # Tab A: 2D Paint & Optimize
        tab_2d = ttk.Frame(self.center_tabs)
        self.center_tabs.add(tab_2d, text="2D Paint & Optimize")

        view_bar = ttk.Frame(tab_2d, style='Panel.TFrame', padding=(8, 4))
        view_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(view_bar, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        view_cb = ttk.Combobox(
            view_bar,
            textvariable=self.var_view_mode,
            values=PaintCanvas.VIEW_MODES,
            state='readonly',
            width=24,
        )
        view_cb.pack(side=tk.LEFT, padx=2)
        view_cb.bind('<<ComboboxSelected>>', lambda _e: self.paint_canvas.set_view_mode(self.var_view_mode.get()))

        ttk.Label(view_bar, text="Mask Overlay:").pack(side=tk.LEFT, padx=(12, 4))
        alpha_scale = ttk.Scale(
            view_bar,
            from_=0.0,
            to=0.85,
            variable=self.var_overlay_alpha,
            command=self._on_alpha_changed,
            length=100,
        )
        alpha_scale.pack(side=tk.LEFT, padx=2)

        ttk.Button(view_bar, text="Fit (F)", command=lambda: self.paint_canvas.fit_to_window()).pack(side=tk.RIGHT, padx=2)
        ttk.Button(view_bar, text="Reset Canvas", command=self._reset_canvas).pack(side=tk.RIGHT, padx=2)
        ttk.Button(view_bar, text="+ Add to Multi-View", command=self.add_current_view_to_multiview).pack(side=tk.RIGHT, padx=2)

        self.paint_canvas = PaintCanvas(
            tab_2d,
            on_color_picked=self._on_color_picked,
            on_status_update=lambda s: self.var_status.set(s),
        )
        self.paint_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

        # Tab B: 3D Interactive Viewport (Arcball Rotation + Direct 3D Mesh Painting)
        tab_3d = ttk.Frame(self.center_tabs)
        self.center_tabs.add(tab_3d, text="3D Interactive Viewport (Arcball & 3D Paint)")

        bar_3d = ttk.Frame(tab_3d, style='Panel.TFrame', padding=(8, 4))
        bar_3d.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(
            bar_3d,
            text="Apply 3D Camera → 2D Render",
            style='Accent.TButton',
            command=self.sync_3d_camera_to_2d,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            bar_3d,
            text="+ Capture 3D View to Multi-View",
            style='Success.TButton',
            command=self.capture_3d_view_to_multiview,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            bar_3d,
            text="Save Painted 3D Texture",
            command=self.save_3d_painted_texture,
        ).pack(side=tk.RIGHT, padx=2)

        self.viewport3d = Viewport3D(
            tab_3d,
            on_color_picked=self._on_color_picked,
            on_texture_modified=self._on_3d_texture_modified,
            on_status_update=lambda s: self.var_status.set(s),
        )
        self.viewport3d.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

        self.select_tool('paint_label')

        # Right Docked Tabbed Inspector
        right = ttk.Frame(body, width=435)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=4)
        right.pack_propagate(False)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_scene_tab()
        self._build_paint_ai_tab()
        self._build_optimize_tab()

        # 3. Bottom Status Bar
        status_bar = ttk.Frame(self, style='Toolbar.TFrame', padding=(8, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self.var_status, style='Status.TLabel').pack(side=tk.LEFT)
        ttk.Label(
            status_bar,
            text="Shortcuts: [B] Brush  [E] Erase  [I] Pick Color  [M] Mask  [X] Erase Mask  [H] Pan  [Ctrl+Z/Y] Undo/Redo  [Wheel] Zoom",
            style='Status.TLabel',
        ).pack(side=tk.RIGHT)

    def _build_tool_strip(self, parent):
        strip = ttk.Frame(parent, style='Panel.TFrame', width=74, padding=(4, 6))
        strip.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=4)
        strip.pack_propagate(False)

        ttk.Label(strip, text="TOOLS", style='Muted.TLabel').pack(pady=(0, 6))
        self.tool_buttons = {}
        tools = [
            ('paint_label', 'Brush\n(B)'),
            ('erase_label', 'Restore\n(E)'),
            ('eyedropper', 'Picker\n(I)'),
            ('paint_mask', 'Mask +\n(M)'),
            ('erase_mask', 'Mask -\n(X)'),
            ('pan', 'Pan\n(H)'),
        ]
        for key, label in tools:
            btn = ttk.Button(strip, text=label, style='Tool.TButton', command=lambda k=key: self.select_tool(k))
            btn.pack(fill=tk.X, pady=2)
            self.tool_buttons[key] = btn

        ttk.Separator(strip, orient='horizontal').pack(fill=tk.X, pady=8)

        # Active brush color preview swatch
        ttk.Label(strip, text="COLOR", style='Muted.TLabel').pack(pady=(0, 3))
        self.color_swatch = tk.Button(
            strip,
            bg='#ff0000',
            activebackground='#ff0000',
            relief='flat',
            height=2,
            command=self.pick_color_dialog,
        )
        self.color_swatch.pack(fill=tk.X, padx=4, pady=2)

        ttk.Separator(strip, orient='horizontal').pack(fill=tk.X, pady=8)
        ttk.Button(strip, text="Undo", style='Tool.TButton', command=lambda: self.paint_canvas.undo()).pack(fill=tk.X, pady=2)
        ttk.Button(strip, text="Redo", style='Tool.TButton', command=lambda: self.paint_canvas.redo()).pack(fill=tk.X, pady=2)
        ttk.Button(strip, text="Clr Mask", style='Tool.TButton', command=lambda: self.paint_canvas.clear_mask()).pack(fill=tk.X, pady=2)

    def _add_file_row(self, parent, row, label, var, filetypes):
        ttk.Label(parent, text=label, width=11).grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(
            parent,
            text="...",
            width=3,
            command=lambda: self._browse_file(var, filetypes),
        ).grid(row=row, column=2, pady=2)

    def _build_scene_tab(self):
        scroll = ScrollableFrame(self.notebook)
        self.notebook.add(scroll, text="1. Scene & Views")
        f = scroll.inner
        f.columnconfigure(0, weight=1)

        # Mesh & Camera Files
        sec1 = ttk.LabelFrame(f, text="Mesh & Camera", padding=8)
        sec1.grid(row=0, column=0, sticky='ew', padx=8, pady=6)
        sec1.columnconfigure(1, weight=1)

        self._add_file_row(sec1, 0, "Mesh (.obj)", self.var_mesh_path, [("OBJ Mesh", "*.obj"), ("All", "*.*")])
        self._add_file_row(sec1, 1, "Texture", self.var_tex_path, [("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        self._add_file_row(sec1, 2, "Intrinsic", self.var_intr_path, [("Text", "*.txt"), ("All", "*.*")])
        self._add_file_row(sec1, 3, "Extrinsic", self.var_extr_path, [("Text", "*.txt"), ("All", "*.*")])

        row_res = ttk.Frame(sec1, style='Panel.TFrame')
        row_res.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(6, 2))
        ttk.Label(row_res, text="Render H:").pack(side=tk.LEFT)
        ttk.Entry(row_res, textvariable=self.var_render_h, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row_res, text="W:").pack(side=tk.LEFT)
        ttk.Entry(row_res, textvariable=self.var_render_w, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row_res, text="Rast:").pack(side=tk.LEFT)
        ttk.Combobox(row_res, textvariable=self.var_rastype, values=('opengl', 'cuda'), state='readonly', width=7).pack(side=tk.LEFT, padx=2)

        row_cam = ttk.Frame(sec1, style='Panel.TFrame')
        row_cam.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(6, 0))
        ttk.Button(row_cam, text="Yaw-", command=lambda: self._nudge_camera(yaw=-15)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cam, text="Yaw+", command=lambda: self._nudge_camera(yaw=15)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cam, text="Zoom+", command=lambda: self._nudge_camera(dz=-0.25)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cam, text="Zoom-", command=lambda: self._nudge_camera(dz=0.25)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cam, text="Reset", command=self._reset_camera).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        ttk.Button(sec1, text="Render Mesh View", style='Accent.TButton', command=self.render_current_view).grid(
            row=6, column=0, columnspan=3, sticky='ew', pady=(8, 0)
        )

        # Multi-Modal 2D Constraints
        sec2 = ttk.LabelFrame(f, text="2D Guidance Constraints (Multi-Modal)", padding=8)
        sec2.grid(row=1, column=0, sticky='ew', padx=8, pady=6)
        sec2.columnconfigure(1, weight=1)

        self._add_file_row(sec2, 0, "Target Label", self.var_label_path, [("Images", "*.png *.jpg"), ("All", "*.*")])
        self._add_file_row(sec2, 1, "Edit Mask", self.var_mask_path, [("Images", "*.png *.jpg"), ("All", "*.*")])
        self._add_file_row(sec2, 2, "Normal Map", self.var_normal_path, [("Normal", "*.npy *.png *.jpg"), ("All", "*.*")])
        self._add_file_row(sec2, 3, "Depth Map", self.var_depth_path, [("Depth", "*.npy *.png"), ("All", "*.*")])
        self._add_file_row(sec2, 4, "Silhouette", self.var_sil_path, [("Mask", "*.png *.jpg"), ("All", "*.*")])

        btn_row = ttk.Frame(sec2, style='Panel.TFrame')
        btn_row.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        ttk.Button(btn_row, text="Load Selected Guidance Maps", command=self.load_guidance_files).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3)
        )
        ttk.Button(btn_row, text="Clear Extra Maps", command=self._clear_extra_guidance).pack(
            side=tk.LEFT, padx=(3, 0)
        )

        # Multi-View Simultaneous Camera Manager
        sec3 = ttk.LabelFrame(f, text="Multi-View Simultaneous Optimization", padding=8)
        sec3.grid(row=2, column=0, sticky='ew', padx=8, pady=(6, 10))
        sec3.columnconfigure(0, weight=1)

        mv_top = ttk.Frame(sec3, style='Panel.TFrame')
        mv_top.grid(row=0, column=0, sticky='ew', pady=(0, 4))
        ttk.Checkbutton(
            mv_top,
            text="Optimize all views simultaneously",
            variable=self.var_use_multiview,
        ).pack(side=tk.LEFT)
        ttk.Entry(mv_top, textvariable=self.var_view_weight, width=5).pack(side=tk.RIGHT)
        ttk.Label(mv_top, text="Weight:").pack(side=tk.RIGHT, padx=(4, 2))

        self.mv_listbox = tk.Listbox(
            sec3,
            height=5,
            bg=BG_INPUT,
            fg=FG_TEXT,
            selectbackground=ACCENT,
            selectforeground='#ffffff',
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=('Consolas', 8),
        )
        self.mv_listbox.grid(row=1, column=0, sticky='ew', pady=4)
        self.mv_listbox.bind('<Double-Button-1>', lambda _e: self.load_selected_multiview())

        mv_btns1 = ttk.Frame(sec3, style='Panel.TFrame')
        mv_btns1.grid(row=2, column=0, sticky='ew', pady=2)
        ttk.Button(mv_btns1, text="+ Add View", style='Accent.TButton', command=self.add_current_view_to_multiview).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )
        ttk.Button(mv_btns1, text="Update", command=self.update_selected_multiview).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )
        ttk.Button(mv_btns1, text="Load", command=self.load_selected_multiview).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )

        mv_btns2 = ttk.Frame(sec3, style='Panel.TFrame')
        mv_btns2.grid(row=3, column=0, sticky='ew', pady=2)
        ttk.Button(mv_btns2, text="Auto 3-View (-25°/0°/+25°)", command=self.generate_turntable_views).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )
        ttk.Button(mv_btns2, text="Remove", command=self.remove_selected_multiview).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(mv_btns2, text="Clear", command=self.clear_multiview_list).pack(
            side=tk.LEFT, padx=1
        )

        ttk.Label(
            sec3,
            text="Tip: Rotate in 3D Viewport → '+ Capture 3D View' or paint multiple views here, then Run Optimization.",
            style='Muted.TLabel',
            wraplength=340,
        ).grid(row=4, column=0, sticky='w', pady=(4, 0))

    def _build_paint_ai_tab(self):
        scroll = ScrollableFrame(self.notebook)
        self.notebook.add(scroll, text="2. Paint & AI")
        f = scroll.inner
        f.columnconfigure(0, weight=1)

        # Brush & Label Settings
        sec1 = ttk.LabelFrame(f, text="Brush & Label Painting", padding=8)
        sec1.grid(row=0, column=0, sticky='ew', padx=8, pady=6)
        sec1.columnconfigure(1, weight=1)

        ttk.Label(sec1, text="Brush Size:").grid(row=0, column=0, sticky='w')
        self.lbl_brush_val = ttk.Label(sec1, text="12 px", width=6)
        self.lbl_brush_val.grid(row=0, column=2, sticky='e')
        ttk.Scale(
            sec1,
            from_=1,
            to=80,
            variable=self.var_brush_radius,
            command=self._on_brush_radius_changed,
        ).grid(row=0, column=1, sticky='ew', padx=6)

        ttk.Checkbutton(
            sec1,
            text="Auto-mark edit mask while painting",
            variable=self.var_auto_mark_mask,
            command=lambda: setattr(self.paint_canvas, 'auto_mark_mask', self.var_auto_mark_mask.get()),
        ).grid(row=1, column=0, columnspan=3, sticky='w', pady=(6, 4))

        # Quick color swatches
        sw_row = ttk.Frame(sec1, style='Panel.TFrame')
        sw_row.grid(row=2, column=0, columnspan=3, sticky='ew', pady=4)
        ttk.Label(sw_row, text="Swatches:").pack(side=tk.LEFT, padx=(0, 6))
        presets = [
            ('#ff0000', (0, 0, 255)),
            ('#00ff00', (0, 255, 0)),
            ('#0066ff', (255, 102, 0)),
            ('#ffff00', (0, 255, 255)),
            ('#ff00ff', (255, 0, 255)),
            ('#ffffff', (255, 255, 255)),
            ('#000000', (0, 0, 0)),
        ]
        for hex_col, bgr in presets:
            tk.Button(
                sw_row,
                bg=hex_col,
                activebackground=hex_col,
                width=2,
                relief='flat',
                command=lambda c=bgr: self._set_brush_color_bgr(c),
            ).pack(side=tk.LEFT, padx=2)
        ttk.Button(sw_row, text="Pick...", command=self.pick_color_dialog).pack(side=tk.RIGHT)

        # Auto Mask Generator
        sec2 = ttk.LabelFrame(f, text="Auto Union Mask (Rendered ∪ Painted)", padding=8)
        sec2.grid(row=1, column=0, sticky='ew', padx=8, pady=6)
        sec2.columnconfigure(1, weight=1)

        ttk.Label(sec2, text="Label Color Tol:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_color_tol, width=8).grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(sec2, text="Diff Thresh:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_diff_thresh, width=8).grid(row=1, column=1, sticky='w', padx=4)
        ttk.Label(sec2, text="Dilate (px):").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_dilate_px, width=8).grid(row=2, column=1, sticky='w', padx=4)

        ttk.Button(sec2, text="Generate Union Mask", style='Accent.TButton', command=self.generate_auto_mask).grid(
            row=3, column=0, columnspan=2, sticky='ew', pady=(6, 0)
        )

        # AI 2D Image Editing Integration
        sec3 = ttk.LabelFrame(f, text="AI 2D Image Editing Backend", padding=8)
        sec3.grid(row=2, column=0, sticky='ew', padx=8, pady=6)
        sec3.columnconfigure(1, weight=1)

        ttk.Label(sec3, text="Backend:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Combobox(
            sec3,
            textvariable=self.var_ai_backend,
            values=AIImageEditor.BACKENDS,
            state='readonly',
        ).grid(row=0, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(sec3, text="Prompt:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(sec3, textvariable=self.var_ai_prompt).grid(row=1, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(sec3, text="Neg Prompt:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(sec3, textvariable=self.var_ai_neg_prompt).grid(row=2, column=1, sticky='ew', padx=4, pady=2)

        p_grid = ttk.Frame(sec3, style='Panel.TFrame')
        p_grid.grid(row=3, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Label(p_grid, text="Strength:").pack(side=tk.LEFT)
        ttk.Entry(p_grid, textvariable=self.var_ai_strength, width=5).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(p_grid, text="CFG:").pack(side=tk.LEFT)
        ttk.Entry(p_grid, textvariable=self.var_ai_cfg, width=5).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(p_grid, text="Steps:").pack(side=tk.LEFT)
        ttk.Entry(p_grid, textvariable=self.var_ai_steps, width=4).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(p_grid, text="Seed:").pack(side=tk.LEFT)
        ttk.Entry(p_grid, textvariable=self.var_ai_seed, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(sec3, text="API URL:").grid(row=4, column=0, sticky='w', pady=2)
        ttk.Entry(sec3, textvariable=self.var_ai_endpoint).grid(row=4, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(sec3, text="HF Model ID:").grid(row=5, column=0, sticky='w', pady=2)
        ttk.Entry(sec3, textvariable=self.var_ai_model_id).grid(row=5, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(sec3, text="API Key:").grid(row=6, column=0, sticky='w', pady=2)
        ttk.Entry(sec3, textvariable=self.var_ai_api_key, show='*').grid(row=6, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(
            sec3,
            text="Tip: For 'local_refine', try prompts: smooth, erode/thin, dilate/thick, sharpen, inpaint.",
            style='Muted.TLabel',
            wraplength=340,
        ).grid(row=7, column=0, columnspan=2, sticky='w', pady=(4, 6))

        self.btn_run_ai = ttk.Button(sec3, text="Run AI 2D Edit", style='Success.TButton', command=self.run_ai_edit)
        self.btn_run_ai.grid(row=8, column=0, columnspan=2, sticky='ew')

    def _build_optimize_tab(self):
        scroll = ScrollableFrame(self.notebook)
        self.notebook.add(scroll, text="3. Optimize & Loss")
        f = scroll.inner
        f.columnconfigure(0, weight=1)

        # Execution controls at top for quick access
        exec_frame = ttk.LabelFrame(f, text="Optimization Runner", padding=8)
        exec_frame.grid(row=0, column=0, sticky='ew', padx=8, pady=6)
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.columnconfigure(1, weight=1)

        self.btn_start_opt = ttk.Button(exec_frame, text="▶ Start Optimization", style='Success.TButton', command=self.start_optimization)
        self.btn_start_opt.grid(row=0, column=0, sticky='ew', padx=(0, 3))
        self.btn_stop_opt = ttk.Button(exec_frame, text="■ Stop", style='Danger.TButton', command=self.stop_optimization, state='disabled')
        self.btn_stop_opt.grid(row=0, column=1, sticky='ew', padx=(3, 0))

        ttk.Progressbar(exec_frame, variable=self.var_progress, maximum=100.0).grid(
            row=1, column=0, columnspan=2, sticky='ew', pady=(8, 2)
        )
        ttk.Label(exec_frame, textvariable=self.var_progress_text, style='Muted.TLabel').grid(
            row=2, column=0, columnspan=2, sticky='w'
        )

        self.loss_chart = LossChart(exec_frame, height=120)
        self.loss_chart.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(6, 0))

        # Optimizer & Schedule
        sec1 = ttk.LabelFrame(f, text="Optimizer & Coarse-to-Fine", padding=8)
        sec1.grid(row=1, column=0, sticky='ew', padx=8, pady=6)
        sec1.columnconfigure(1, weight=1)
        sec1.columnconfigure(3, weight=1)

        ttk.Label(sec1, text="Iterations:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_iters, width=8).grid(row=0, column=1, sticky='ew', padx=4)
        ttk.Label(sec1, text="LR:").grid(row=0, column=2, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_lr, width=8).grid(row=0, column=3, sticky='ew', padx=4)

        ttk.Label(sec1, text="LR Decay:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_lr_decay, width=8).grid(row=1, column=1, sticky='ew', padx=4)
        ttk.Label(sec1, text="Log Every:").grid(row=1, column=2, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_log_interval, width=8).grid(row=1, column=3, sticky='ew', padx=4)

        ttk.Label(sec1, text="Save Img:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_save_interval, width=8).grid(row=2, column=1, sticky='ew', padx=4)
        ttk.Label(sec1, text="Save Mesh:").grid(row=2, column=2, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_save_mesh_interval, width=8).grid(row=2, column=3, sticky='ew', padx=4)

        ttk.Label(sec1, text="Output Dir:").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(sec1, textvariable=self.var_out_dir).grid(row=3, column=1, columnspan=3, sticky='ew', padx=4, pady=2)

        blur_row = ttk.Frame(sec1, style='Panel.TFrame')
        blur_row.grid(row=4, column=0, columnspan=4, sticky='ew', pady=(4, 0))
        ttk.Checkbutton(blur_row, text="Coarse-to-Fine Gaussian Blur", variable=self.var_use_blur).pack(side=tk.LEFT)
        ttk.Label(blur_row, text="Kernel:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(blur_row, textvariable=self.var_blur_k, width=4).pack(side=tk.LEFT)

        # Automatic Loss Adaptation
        sec2 = ttk.LabelFrame(f, text="Automatic Loss Adaptation", padding=8)
        sec2.grid(row=2, column=0, sticky='ew', padx=8, pady=6)
        sec2.columnconfigure(1, weight=1)
        sec2.columnconfigure(3, weight=1)

        ttk.Label(sec2, text="Strategy:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Combobox(
            sec2,
            textvariable=self.var_adaptive_mode,
            values=('fixed', 'ema_scale', 'grad_norm', 'dwa'),
            state='readonly',
        ).grid(row=0, column=1, columnspan=3, sticky='ew', padx=4, pady=2)

        ttk.Label(sec2, text="Interval:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_adapt_interval, width=6).grid(row=1, column=1, sticky='ew', padx=4)
        ttk.Label(sec2, text="EMA:").grid(row=1, column=2, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_adapt_ema, width=6).grid(row=1, column=3, sticky='ew', padx=4)

        ttk.Label(sec2, text="Min Scale:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_adapt_min, width=6).grid(row=2, column=1, sticky='ew', padx=4)
        ttk.Label(sec2, text="Max Scale:").grid(row=2, column=2, sticky='w', pady=2)
        ttk.Entry(sec2, textvariable=self.var_adapt_max, width=6).grid(row=2, column=3, sticky='ew', padx=4)

        # Loss Function Weights
        sec3 = ttk.LabelFrame(f, text="Loss Function Weights", padding=8)
        sec3.grid(row=3, column=0, sticky='ew', padx=8, pady=6)
        sec3.columnconfigure(1, weight=1)
        sec3.columnconfigure(3, weight=1)

        preset_row = ttk.Frame(sec3, style='Panel.TFrame')
        preset_row.grid(row=0, column=0, columnspan=4, sticky='ew', pady=(0, 6))
        ttk.Label(preset_row, text="Preset:").pack(side=tk.LEFT)
        preset_cb = ttk.Combobox(
            preset_row,
            textvariable=self.var_loss_preset,
            values=list(LOSS_PRESETS.keys()),
            state='readonly',
            width=26,
        )
        preset_cb.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        preset_cb.bind('<<ComboboxSelected>>', self._apply_loss_preset)

        labels = [
            ('img_loss', 'Img L1'),
            ('img_l2_loss', 'Img L2'),
            ('ssim_loss', 'SSIM'),
            ('vgg_loss', 'VGG Perc'),
            ('normal_loss', 'Normal Map'),
            ('depth_loss', 'Depth Map'),
            ('silhouette_loss', 'Silhouette'),
            ('nc_loss', 'Normal Cons'),
            ('edge_loss', 'Edge Len'),
            ('area_loss', 'Face Area'),
            ('lap_loss', 'Laplacian'),
            ('diff_lap_loss', 'Diff Lap'),
            ('disp_loss', 'Anchor Disp'),
        ]
        for idx, (k, title) in enumerate(labels):
            r = 1 + idx // 2
            c = (idx % 2) * 2
            ttk.Label(sec3, text=title, font=('Segoe UI', 8)).grid(row=r, column=c, sticky='w', pady=2)
            ttk.Entry(sec3, textvariable=self.loss_vars[k], width=9).grid(row=r, column=c + 1, sticky='ew', padx=(2, 6), pady=2)

        # Console Log
        sec4 = ttk.LabelFrame(f, text="Console Log", padding=6)
        sec4.grid(row=4, column=0, sticky='ew', padx=8, pady=(6, 10))
        self.log_text = tk.Text(
            sec4,
            height=7,
            bg='#141519',
            fg='#cbd5e1',
            insertbackground='#ffffff',
            font=('Consolas', 8),
            relief='flat',
            wrap='word',
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _bind_shortcuts(self):
        self.bind('<Control-z>', lambda _e: self.paint_canvas.undo())
        self.bind('<Control-y>', lambda _e: self.paint_canvas.redo())

        def _tool_shortcut(tool_name):
            def handler(event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                    return
                self.select_tool(tool_name)
            return handler

        self.bind('b', _tool_shortcut('paint_label'))
        self.bind('e', _tool_shortcut('erase_label'))
        self.bind('i', _tool_shortcut('eyedropper'))
        self.bind('m', _tool_shortcut('paint_mask'))
        self.bind('x', _tool_shortcut('erase_mask'))
        self.bind('h', _tool_shortcut('pan'))

        def _fit_shortcut(event):
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                return
            self.paint_canvas.fit_to_window()

        self.bind('f', _fit_shortcut)

        def _vp3d_shortcut(mode_name):
            def handler(event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                    return
                self.viewport3d.set_mode(mode_name)
            return handler

        self.bind('o', _vp3d_shortcut('orbit'))
        self.bind('p', _vp3d_shortcut('paint3d'))
        self.bind('r', _vp3d_shortcut('erase3d'))
        self.bind('k', _vp3d_shortcut('pick3d'))

    # ------------------------------------------------------------------
    # Helpers & Callbacks
    # ------------------------------------------------------------------

    def log(self, msg):
        def _append():
            self.log_text.insert(tk.END, msg + '\n')
            self.log_text.see(tk.END)
        self.after(0, _append)

    def _browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def select_tool(self, tool_name):
        self.paint_canvas.set_tool(tool_name)
        for k, btn in self.tool_buttons.items():
            btn.configure(style='ToolActive.TButton' if k == tool_name else 'Tool.TButton')

    def _set_brush_color_bgr(self, bgr):
        col = tuple(int(c) for c in bgr)
        self.paint_canvas.brush_color_bgr = col
        if hasattr(self, 'viewport3d'):
            self.viewport3d.brush_color_bgr = col
        r, g, b = col[2], col[1], col[0]
        hex_col = f'#{r:02x}{g:02x}{b:02x}'
        self.color_swatch.configure(bg=hex_col, activebackground=hex_col)

    def _on_color_picked(self, bgr):
        self._set_brush_color_bgr(bgr)
        self.select_tool('paint_label')
        self.log(f"Sampled color BGR={bgr}")

    def pick_color_dialog(self):
        b, g, r = self.paint_canvas.brush_color_bgr
        chosen = colorchooser.askcolor(initialcolor=f'#{r:02x}{g:02x}{b:02x}')
        if chosen and chosen[0]:
            rr, gg, bb = [int(round(v)) for v in chosen[0]]
            self._set_brush_color_bgr((bb, gg, rr))

    def _on_brush_radius_changed(self, val):
        r = max(1, int(round(float(val))))
        self.paint_canvas.brush_radius = r
        if hasattr(self, 'viewport3d'):
            self.viewport3d.brush_radius = r
        self.lbl_brush_val.configure(text=f"{r} px")

    def _on_alpha_changed(self, val):
        self.paint_canvas.overlay_alpha = float(val)
        self.paint_canvas.refresh()

    def _apply_loss_preset(self, _event=None):
        name = self.var_loss_preset.get()
        preset = LOSS_PRESETS.get(name)
        if not preset:
            return
        for k, v in preset.items():
            if k in self.loss_vars:
                self.loss_vars[k].set(v)
        self.log(f"Applied loss preset: {name}")

    def _reset_canvas(self):
        self.paint_canvas.reset_label_to_render()
        self.log("Reset target label to rendered view.")

    # ------------------------------------------------------------------
    # Sample Loading & Camera Rendering
    # ------------------------------------------------------------------

    def load_sample_preset(self, preset_name):
        root = os.getcwd()
        if preset_name == 'cow':
            self.var_mesh_path.set(os.path.join(root, 'sample_data', 'mesh', 'cow.obj'))
            tex = os.path.join(root, 'sample_data', 'mesh', 'cow_texture_with_label.png')
            if not os.path.exists(tex):
                tex = os.path.join(root, 'sample_data', 'mesh', 'cow_texture.png')
            self.var_tex_path.set(tex)
            self.var_intr_path.set(os.path.join(root, 'sample_data', 'intr.txt'))
            self.var_extr_path.set(os.path.join(root, 'sample_data', 'extr.txt'))
            self.var_label_path.set(os.path.join(root, 'sample_data', 'label.png'))
            self.var_mask_path.set(os.path.join(root, 'sample_data', 'mask.png'))
            self.var_normal_path.set('')
            self.var_depth_path.set('')
            self.var_sil_path.set('')
            self.var_render_h.set(1024)
            self.var_render_w.set(1024)
        elif preset_name == 'woman':
            self.var_mesh_path.set(os.path.join(root, 'sample_data', 'woman', 'mesh', 'texture_manif.obj'))
            self.var_tex_path.set(os.path.join(root, 'sample_data', 'woman', 'mesh', 'texture_material0000_map_kd.png'))
            self.var_intr_path.set(os.path.join(root, 'sample_data', 'woman', 'intr.txt'))
            self.var_extr_path.set(os.path.join(root, 'sample_data', 'woman', 'extr.txt'))
            self.var_label_path.set(os.path.join(root, 'sample_data', 'woman', 'label.png'))
            self.var_mask_path.set(os.path.join(root, 'sample_data', 'woman', 'mask.png'))
            self.var_normal_path.set(os.path.join(root, 'sample_data', 'woman', 'normal.npy'))
            self.var_depth_path.set('')
            self.var_sil_path.set('')
            self.var_render_h.set(1792)
            self.var_render_w.set(828)

        self.intrinsic = None
        self.extrinsic = None
        self.base_extrinsic = None
        self.clear_multiview_list(silent=True)
        self.log(f"Loaded preset paths: {preset_name}")
        self.render_current_view(load_existing_guidance=True, reload_3d=True, camera_changed=True)

    def _ensure_camera_loaded(self):
        intr_path = self.var_intr_path.get().strip()
        extr_path = self.var_extr_path.get().strip()
        if not intr_path or not os.path.exists(intr_path):
            raise FileNotFoundError("Please specify a valid camera intrinsic (.txt) file.")
        if not extr_path or not os.path.exists(extr_path):
            raise FileNotFoundError("Please specify a valid camera extrinsic (.txt) file.")

        if self.intrinsic is None:
            self.intrinsic = np.loadtxt(intr_path).reshape(4, 4).astype(np.float32)
        if self.extrinsic is None:
            self.extrinsic = np.loadtxt(extr_path).reshape(4, 4).astype(np.float32)
            self.base_extrinsic = self.extrinsic.copy()

    def _nudge_camera(self, yaw=0.0, dz=0.0):
        try:
            self._ensure_camera_loaded()
        except Exception as exc:
            messagebox.showwarning("Camera", str(exc))
            return

        if abs(yaw) > 1e-6:
            rad = np.deg2rad(yaw)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            # Orbit around camera target
            self.extrinsic = self.extrinsic @ rot
        if abs(dz) > 1e-6:
            self.extrinsic[2, 3] += float(dz)

        self.render_current_view(load_existing_guidance=False, camera_changed=True)

    def _reset_camera(self):
        if self.base_extrinsic is not None:
            self.extrinsic = self.base_extrinsic.copy()
            self.render_current_view(load_existing_guidance=False, camera_changed=True)

    def render_current_view(self, load_existing_guidance=False, reload_3d=False, camera_changed=False):
        obj_path = self.var_mesh_path.get().strip()
        tex_path = self.var_tex_path.get().strip()
        if not obj_path or not os.path.exists(obj_path):
            messagebox.showwarning("Mesh Required", "Please select a valid .obj mesh file first.")
            return

        try:
            self._ensure_camera_loaded()
            h, w = int(self.var_render_h.get()), int(self.var_render_w.get())

            # If user painted directly in the 3D viewport, persist live texture before rendering
            active_tex_path = tex_path if (tex_path and os.path.exists(tex_path)) else None
            if hasattr(self, 'viewport3d') and self.viewport3d._texture_dirty and self.viewport3d.tex_bgr is not None:
                live_tex = os.path.join(self.var_out_dir.get().strip() or './output', '_live_3d_painted_tex.png')
                self.viewport3d.save_painted_texture(live_tex)
                active_tex_path = live_tex
                self.var_tex_path.set(live_tex)

            self.log(f"Rendering {os.path.basename(obj_path)} at {h}x{w}...")
            res = render_mesh_view(
                obj_path=obj_path,
                tex_path=active_tex_path,
                intrinsic=self.intrinsic,
                extrinsic=self.extrinsic,
                render_size=(h, w),
                device=self.var_device.get(),
                rastype=self.var_rastype.get(),
                prefer_gpu=self.gpu_ok,
            )
        except Exception as exc:
            messagebox.showerror("Render Error", str(exc))
            self.log(f"[Error] Render failed: {exc}")
            return

        self.last_render_fg_mask = res.get('mask_gray')
        self.paint_canvas.set_rendered_image(
            res['color_bgr'],
            reset_label=True,
            reset_mask=camera_changed,
            fit=True,
        )
        if 'normal' in res and res['normal'] is not None:
            self.paint_canvas.set_normal_vis(vis_normal_map(res['normal']))
        if 'depth' in res and res['depth'] is not None:
            self.paint_canvas.set_depth_vis(vis_depth_map(res['depth']))
        if 'mesh' in res:
            self.current_mesh = res['mesh']

        # When camera viewpoint changes, clear stale view-specific guidance maps and
        # project any direct 3D painted UV region accurately into the new 2D camera view.
        if camera_changed and not load_existing_guidance:
            self.normal_target = None
            self.depth_target = None
            self.silhouette_target = None
            if hasattr(self, 'viewport3d') and self.viewport3d.has_3d_paint_edits():
                proj_mask = self.viewport3d.project_3d_paint_mask(self.intrinsic, self.extrinsic, (h, w))
                if proj_mask is not None:
                    self.paint_canvas.set_mask_image(proj_mask, push_history=False)
                    cov = 100.0 * float((proj_mask > 127).mean())
                    self.log(f"Projected 3D painted region to new 2D mask ({cov:.2f}% of image).")
            else:
                self.log("Camera viewpoint changed: cleared previous view's 2D mask (paint or click 'Auto Mask' for this angle).")

        # Keep 3D Interactive Viewport synchronized with current mesh & camera
        if hasattr(self, 'viewport3d'):
            if reload_3d or self.viewport3d.verts is None or self.viewport3d.obj_path != obj_path:
                try:
                    self.viewport3d.load_mesh(
                        obj_path=obj_path,
                        tex_path=active_tex_path,
                        intrinsic=self.intrinsic,
                        extrinsic=self.extrinsic,
                        render_size=(h, w),
                    )
                except Exception as exc:
                    self.log(f"[Warn] 3D Viewport load failed: {exc}")
            else:
                self.viewport3d.set_extrinsic(self.extrinsic)

        self.log(f"Render complete ({res['backend']}).")
        if load_existing_guidance:
            self.load_guidance_files(silent=True)

    def load_guidance_files(self, silent=False):
        h, w = int(self.var_render_h.get()), int(self.var_render_w.get())
        loaded = []
        try:
            lp = self.var_label_path.get().strip()
            if lp and os.path.exists(lp):
                img = cv2.imread(lp, cv2.IMREAD_COLOR)
                if img is not None:
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                    self.paint_canvas.set_label_image(img, push_history=False)
                    loaded.append('label')

            mp = self.var_mask_path.get().strip()
            if mp and os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.paint_canvas.set_mask_image(m, push_history=False)
                    loaded.append('mask')

            np_path = self.var_normal_path.get().strip()
            if np_path and os.path.exists(np_path):
                self.normal_target = load_normal_map_file(np_path, target_size=(h, w))
                self.paint_canvas.set_normal_vis(vis_normal_map(self.normal_target))
                loaded.append('normal')
            else:
                self.normal_target = None

            dp_path = self.var_depth_path.get().strip()
            if dp_path and os.path.exists(dp_path):
                self.depth_target = load_depth_map_file(dp_path, target_size=(h, w))
                self.paint_canvas.set_depth_vis(vis_depth_map(self.depth_target))
                loaded.append('depth')
            else:
                self.depth_target = None

            sp_path = self.var_sil_path.get().strip()
            if sp_path and os.path.exists(sp_path):
                s = cv2.imread(sp_path, cv2.IMREAD_GRAYSCALE)
                if s is not None:
                    if s.shape[:2] != (h, w):
                        s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.silhouette_target = (s > 127).astype(np.float32)
                    loaded.append('silhouette')
            else:
                self.silhouette_target = None
        except Exception as exc:
            if not silent:
                messagebox.showerror("Load Guidance Error", str(exc))
            self.log(f"[Error] Loading guidance maps: {exc}")
            return

        if loaded:
            self.log(f"Loaded guidance maps: {', '.join(loaded)}")
        elif not silent:
            messagebox.showinfo("Guidance Maps", "No existing guidance map files were selected.")

    def _clear_extra_guidance(self):
        self.var_normal_path.set('')
        self.var_depth_path.set('')
        self.var_sil_path.set('')
        self.normal_target = None
        self.depth_target = None
        self.silhouette_target = None
        self.log("Cleared normal, depth, and silhouette targets.")

    # ------------------------------------------------------------------
    # Auto Mask & AI 2D Editing
    # ------------------------------------------------------------------

    def generate_auto_mask(self):
        if self.paint_canvas.label_bgr is None:
            messagebox.showwarning("No Image", "Render or load an image first.")
            return

        mask = compute_edit_mask(
            rendered_bgr=self.paint_canvas.rendered_bgr,
            edited_bgr=self.paint_canvas.label_bgr,
            painted_mask=self.paint_canvas.mask_gray,
            label_color_bgr=self.paint_canvas.brush_color_bgr,
            color_tol=self.var_color_tol.get(),
            diff_thresh=self.var_diff_thresh.get(),
            dilate_px=self.var_dilate_px.get(),
            fg_mask=self.last_render_fg_mask,
        )
        self.paint_canvas.set_mask_image(mask, push_history=True)
        coverage = 100.0 * float((mask > 127).mean())
        self.log(f"Generated union edit mask ({coverage:.2f}% of image).")

    def run_ai_edit(self):
        if self._ai_busy:
            return
        if self.paint_canvas.label_bgr is None:
            messagebox.showwarning("No Image", "Render or load an image before running AI 2D editing.")
            return

        src_img = self.paint_canvas.label_bgr.copy()
        cur_mask = self.paint_canvas.mask_gray.copy() if self.paint_canvas.mask_gray is not None else None
        if cur_mask is not None and not np.any(cur_mask > 127):
            cur_mask = None  # Edit entire view if user hasn't painted a localized mask

        params = {
            'backend': self.var_ai_backend.get(),
            'prompt': self.var_ai_prompt.get(),
            'negative_prompt': self.var_ai_neg_prompt.get(),
            'strength': float(self.var_ai_strength.get()),
            'guidance_scale': float(self.var_ai_cfg.get()),
            'steps': int(self.var_ai_steps.get()),
            'endpoint': self.var_ai_endpoint.get().strip(),
            'model_id': self.var_ai_model_id.get().strip() or None,
            'api_key': self.var_ai_api_key.get().strip(),
            'device': self.var_device.get(),
            'seed': int(self.var_ai_seed.get()),
        }

        self._ai_busy = True
        self.btn_run_ai.configure(state='disabled', text='Running AI Edit...')
        self.log(f"Running AI 2D Edit ({params['backend']}) with prompt: '{params['prompt']}'")

        def _worker():
            try:
                edited_bgr, updated_mask = self.ai_editor.edit(src_img, mask_gray=cur_mask, **params)
                def _apply():
                    self.paint_canvas.set_label_image(edited_bgr, push_history=True)
                    self.paint_canvas.set_mask_image(updated_mask, push_history=False)
                    self.log("AI 2D Edit applied (Undo with Ctrl+Z).")
                self.after(0, _apply)
            except Exception as exc:
                self.log(f"[Error] AI 2D Edit failed: {exc}")
                self.after(0, lambda e=str(exc): messagebox.showerror("AI 2D Edit Error", e))
            finally:
                def _done():
                    self._ai_busy = False
                    self.btn_run_ai.configure(state='normal', text='Run AI 2D Edit')
                self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Save / Export
    # ------------------------------------------------------------------

    def save_label_and_mask(self):
        if self.paint_canvas.label_bgr is None:
            messagebox.showwarning("Nothing to Save", "Render or load an image first.")
            return
        out_dir = filedialog.askdirectory(initialdir=self.var_out_dir.get() or '.')
        if not out_dir:
            return
        label_file = os.path.join(out_dir, 'label_edited.png')
        mask_file = os.path.join(out_dir, 'mask_edited.png')
        cv2.imwrite(label_file, self.paint_canvas.label_bgr)
        if self.paint_canvas.mask_gray is not None:
            cv2.imwrite(mask_file, self.paint_canvas.mask_gray)
        self.log(f"Saved edited label & mask to: {out_dir}")
        messagebox.showinfo("Saved", f"Saved:\n{label_file}\n{mask_file}")

    def export_deformed_mesh(self):
        if self.current_mesh is None:
            messagebox.showwarning("No Mesh", "Run optimization (or GPU render) first so a deformed mesh is available.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension='.obj',
            filetypes=[("OBJ Mesh", "*.obj")],
            initialfile='deformed.obj',
            initialdir=self.var_out_dir.get() or '.',
        )
        if not save_path:
            return
        try:
            from util.obj_io import save_obj
            save_obj(save_path, self.current_mesh)
            self.log(f"Exported mesh to: {save_path}")
            messagebox.showinfo("Exported", f"Mesh exported to:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ------------------------------------------------------------------
    # Multi-View Manager & 3D Viewport Synchronization
    # ------------------------------------------------------------------

    def _refresh_multiview_listbox(self):
        self.mv_listbox.delete(0, tk.END)
        for idx, v in enumerate(self.multiview_list):
            mask = v.get('mask_gray')
            cov = 100.0 * float((mask > 127).mean()) if mask is not None else 0.0
            tags = []
            if v.get('normal_target') is not None:
                tags.append('N')
            if v.get('depth_target') is not None:
                tags.append('D')
            if v.get('silhouette_target') is not None:
                tags.append('S')
            tag_str = f" [{'+'.join(tags)}]" if tags else ""
            marker = "▸ " if idx == self._active_view_idx else "  "
            self.mv_listbox.insert(
                tk.END,
                f"{marker}{idx + 1}. {v['name']} (w={v['weight']:.1f}, mask={cov:.1f}%){tag_str}",
            )

    def _snapshot_current_view(self, name=None):
        self._ensure_camera_loaded()
        if self.paint_canvas.label_bgr is None:
            raise ValueError("Render or load a view before adding to Multi-View list.")
        return {
            'name': name or f"View {len(self.multiview_list) + 1}",
            'intrinsic': self.intrinsic.copy(),
            'extrinsic': self.extrinsic.copy(),
            'render_size': (int(self.var_render_h.get()), int(self.var_render_w.get())),
            'rendered_bgr': self.paint_canvas.rendered_bgr.copy() if self.paint_canvas.rendered_bgr is not None else None,
            'label_bgr': self.paint_canvas.label_bgr.copy(),
            'mask_gray': self.paint_canvas.mask_gray.copy() if self.paint_canvas.mask_gray is not None else None,
            'normal_target': self.normal_target.copy() if self.normal_target is not None else None,
            'depth_target': self.depth_target.copy() if self.depth_target is not None else None,
            'silhouette_target': self.silhouette_target.copy() if self.silhouette_target is not None else None,
            'weight': max(0.01, float(self.var_view_weight.get())),
        }

    def add_current_view_to_multiview(self, name=None):
        try:
            snap = self._snapshot_current_view(name=name)
        except Exception as exc:
            messagebox.showwarning("Add View", str(exc))
            return
        self.multiview_list.append(snap)
        self._active_view_idx = len(self.multiview_list) - 1
        self._refresh_multiview_listbox()
        self.log(f"Added '{snap['name']}' to Multi-View list ({len(self.multiview_list)} views total).")

    def update_selected_multiview(self):
        sel = self.mv_listbox.curselection()
        idx = sel[0] if sel else self._active_view_idx
        if idx is None or not (0 <= idx < len(self.multiview_list)):
            messagebox.showwarning("Update View", "Select a view from the Multi-View list first.")
            return
        try:
            snap = self._snapshot_current_view(name=self.multiview_list[idx]['name'])
        except Exception as exc:
            messagebox.showwarning("Update View", str(exc))
            return
        self.multiview_list[idx] = snap
        self._active_view_idx = idx
        self._refresh_multiview_listbox()
        self.log(f"Updated multi-view item #{idx + 1} ('{snap['name']}').")

    def load_selected_multiview(self):
        sel = self.mv_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if not (0 <= idx < len(self.multiview_list)):
            return
        v = self.multiview_list[idx]
        self._active_view_idx = idx
        self.intrinsic = v['intrinsic'].copy()
        self.extrinsic = v['extrinsic'].copy()
        self.var_render_h.set(v['render_size'][0])
        self.var_render_w.set(v['render_size'][1])
        self.var_view_weight.set(v['weight'])

        if v['rendered_bgr'] is not None:
            self.paint_canvas.set_rendered_image(v['rendered_bgr'], reset_label=False, fit=True)
        if v['label_bgr'] is not None:
            self.paint_canvas.set_label_image(v['label_bgr'], push_history=False)
        if v['mask_gray'] is not None:
            self.paint_canvas.set_mask_image(v['mask_gray'], push_history=False)

        self.normal_target = v['normal_target'].copy() if v['normal_target'] is not None else None
        self.depth_target = v['depth_target'].copy() if v['depth_target'] is not None else None
        self.silhouette_target = v['silhouette_target'].copy() if v['silhouette_target'] is not None else None
        if hasattr(self, 'viewport3d') and self.viewport3d.verts is not None:
            self.viewport3d.set_extrinsic(self.extrinsic)

        self._refresh_multiview_listbox()
        self.log(f"Loaded multi-view item #{idx + 1} ('{v['name']}').")

    def remove_selected_multiview(self):
        sel = self.mv_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        removed = self.multiview_list.pop(idx)
        self._active_view_idx = None
        self._refresh_multiview_listbox()
        self.log(f"Removed '{removed['name']}' from Multi-View list.")

    def clear_multiview_list(self, silent=False):
        self.multiview_list.clear()
        self._active_view_idx = None
        if hasattr(self, 'mv_listbox'):
            self._refresh_multiview_listbox()
        if not silent:
            self.log("Cleared Multi-View list.")

    def generate_turntable_views(self, angles=(-25.0, 0.0, 25.0)):
        obj_path = self.var_mesh_path.get().strip()
        if not obj_path or not os.path.exists(obj_path):
            messagebox.showwarning("Mesh Required", "Load a mesh first before generating multi-view poses.")
            return
        try:
            self._ensure_camera_loaded()
        except Exception as exc:
            messagebox.showerror("Camera Error", str(exc))
            return

        base_e = (self.base_extrinsic if self.base_extrinsic is not None else self.extrinsic).copy()
        for deg in angles:
            rad = np.deg2rad(deg)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            self.extrinsic = base_e @ rot
            self.render_current_view(load_existing_guidance=False, camera_changed=True)
            self.add_current_view_to_multiview(name=f"Yaw {deg:+.0f}°")

        # Restore base view
        self.extrinsic = base_e.copy()
        self.log(f"Generated {len(angles)} turntable views for simultaneous multi-view optimization.")

    def _on_3d_texture_modified(self, _tex_bgr):
        self.log("3D mesh texture updated (click 'Apply 3D Camera → 2D Render' or 'Save Painted 3D Texture' to sync).")

    def sync_3d_camera_to_2d(self):
        if self.viewport3d.verts is None:
            messagebox.showwarning("3D Viewport", "Load a mesh first.")
            return
        self.extrinsic = self.viewport3d.get_extrinsic()
        self.render_current_view(load_existing_guidance=False, camera_changed=True)
        self.center_tabs.select(0)
        self.log("Synced 3D Arcball camera & live texture to 2D Paint view.")

    def capture_3d_view_to_multiview(self):
        if self.viewport3d.verts is None:
            messagebox.showwarning("3D Viewport", "Load a mesh first.")
            return
        self.extrinsic = self.viewport3d.get_extrinsic()
        self.render_current_view(load_existing_guidance=False, camera_changed=True)
        self.add_current_view_to_multiview(name=f"3D Pose #{len(self.multiview_list) + 1}")

    def save_3d_painted_texture(self):
        if self.viewport3d.tex_bgr is None:
            messagebox.showwarning("No Texture", "No UV texture loaded in the 3D Viewport.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[("PNG Texture", "*.png")],
            initialfile='painted_texture.png',
            initialdir=self.var_out_dir.get() or '.',
        )
        if not save_path:
            return
        try:
            self.viewport3d.save_painted_texture(save_path)
            self.var_tex_path.set(save_path)
            self.log(f"Saved 3D painted texture to: {save_path}")
            messagebox.showinfo("Saved", f"Saved painted texture:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Save Texture Error", str(exc))

    # ------------------------------------------------------------------
    # 3D Mesh Optimization Runner
    # ------------------------------------------------------------------

    def start_optimization(self):
        if self._opt_thread is not None and self._opt_thread.is_alive():
            return

        if not self.gpu_ok:
            messagebox.showerror(
                "GPU / Nvdiffrast Required",
                f"Differentiable mesh optimization requires PyTorch + CUDA + nvdiffrast.\n"
                f"Current environment: {self.gpu_status}\n\n"
                f"Tip: You can still paint in 2D/3D, build multi-view targets, run AI 2D edits, "
                f"and save your edited label/mask/texture for a GPU machine.",
            )
            return

        obj_path = self.var_mesh_path.get().strip()
        if not obj_path or not os.path.exists(obj_path):
            messagebox.showwarning("Mesh Required", "Please select a valid .obj mesh file.")
            return

        use_mv = bool(self.var_use_multiview.get() and len(self.multiview_list) >= 2)
        if not use_mv and self.paint_canvas.label_bgr is None:
            messagebox.showwarning("Target Required", "Please render/load and paint a target label image first.")
            return

        try:
            self._ensure_camera_loaded()
        except Exception as exc:
            messagebox.showerror("Camera Error", str(exc))
            return

        # Persist 3D painted texture if dirty so optimizer uses the exact painted mesh texture
        tex_path = self.var_tex_path.get().strip() or None
        if hasattr(self, 'viewport3d') and self.viewport3d._texture_dirty and self.viewport3d.tex_bgr is not None:
            live_tex = os.path.join(self.var_out_dir.get().strip() or './output', '_live_3d_painted_tex.png')
            self.viewport3d.save_painted_texture(live_tex)
            tex_path = live_tex
            self.var_tex_path.set(live_tex)

        mv_snapshots = []
        if use_mv:
            for v in self.multiview_list:
                mv_snapshots.append({
                    'name': v['name'],
                    'intrinsic': v['intrinsic'].copy(),
                    'extrinsic': v['extrinsic'].copy(),
                    'render_size': v['render_size'],
                    'label_bgr': v['label_bgr'].copy() if v['label_bgr'] is not None else None,
                    'mask_gray': v['mask_gray'].copy() if v['mask_gray'] is not None else None,
                    'normal_target': v['normal_target'].copy() if v['normal_target'] is not None else None,
                    'depth_target': v['depth_target'].copy() if v['depth_target'] is not None else None,
                    'silhouette_target': v['silhouette_target'].copy() if v['silhouette_target'] is not None else None,
                    'weight': float(v['weight']),
                })

        # Snapshot all settings from UI thread before starting background thread
        cfg = {
            'obj_path': obj_path,
            'tex_path': tex_path,
            'intrinsic': self.intrinsic.copy(),
            'extrinsic': self.extrinsic.copy(),
            'render_size': (int(self.var_render_h.get()), int(self.var_render_w.get())),
            'device': self.var_device.get(),
            'rastype': self.var_rastype.get(),
            'label_bgr': self.paint_canvas.label_bgr.copy() if self.paint_canvas.label_bgr is not None else None,
            'mask_gray': self.paint_canvas.mask_gray.copy() if self.paint_canvas.mask_gray is not None else None,
            'normal_target': self.normal_target.copy() if self.normal_target is not None else None,
            'depth_target': self.depth_target.copy() if self.depth_target is not None else None,
            'silhouette_target': self.silhouette_target.copy() if self.silhouette_target is not None else None,
            'multiviews': mv_snapshots,
            'out_dir': self.var_out_dir.get().strip() or './output',
            'pme_args': {
                'iteration': int(self.var_iters.get()),
                'lr': float(self.var_lr.get()),
                'lr_decay': float(self.var_lr_decay.get()),
                'log_interval': max(1, int(self.var_log_interval.get())),
                'save_interval': int(self.var_save_interval.get()),
                'save_mesh_interval': int(self.var_save_mesh_interval.get()),
                'display_image': False,
                'USE_BLUR': bool(self.var_use_blur.get()),
                'BLUR_KERNEL_SIZE': int(self.var_blur_k.get()),
                'adaptive_mode': self.var_adaptive_mode.get(),
                'adapt_interval': int(self.var_adapt_interval.get()),
                'adapt_ema': float(self.var_adapt_ema.get()),
                'adapt_min_scale': float(self.var_adapt_min.get()),
                'adapt_max_scale': float(self.var_adapt_max.get()),
                'loss_weights': {k: float(v.get()) for k, v in self.loss_vars.items()},
            },
        }

        self._stop_event.clear()
        self.loss_chart.clear()
        self.var_progress.set(0.0)
        self.var_progress_text.set("Initializing...")
        self.btn_start_opt.configure(state='disabled')
        self.btn_stop_opt.configure(state='normal')
        self.center_tabs.select(0)
        self.var_view_mode.set('Optimization Live')
        self.paint_canvas.set_view_mode('Optimization Live')
        mode_desc = f"{len(mv_snapshots)}-view simultaneous" if mv_snapshots else "single-view"
        self.log(
            f"Starting {mode_desc} PME optimization "
            f"({cfg['pme_args']['iteration']} iters, adaptive={cfg['pme_args']['adaptive_mode']})..."
        )

        self._opt_thread = threading.Thread(target=self._run_optimization_worker, args=(cfg,), daemon=True)
        self._opt_thread.start()

    def stop_optimization(self):
        if self._opt_thread is not None and self._opt_thread.is_alive():
            self._stop_event.set()
            self.var_progress_text.set("Stopping...")
            self.log("Stop requested by user...")

    def _run_optimization_worker(self, cfg):
        try:
            import torch
            from util.obj_io import load_obj, save_obj
            from render.ulit_renderer import NvdiffrastMeshUlitRenderer
            from core.editing import PaintMeshEditing, ViewConstraint

            device = cfg['device']
            mesh = load_obj(cfg['obj_path'], cfg['tex_path'], device)
            mesh.vs.requires_grad_(True)

            renderer = NvdiffrastMeshUlitRenderer({
                'device': device,
                'rastype': cfg['rastype'],
                'render_size': cfg['render_size'],
            })

            def on_step(state):
                pct = 100.0 * state['iter'] / max(1, state['total_iters'])
                preview = state.get('preview_bgr')
                nv = state.get('num_views', 1)
                v_tag = f" ({nv} views)" if nv > 1 else ""
                summary = f"Iter {state['iter']}/{state['total_iters']}{v_tag}  |  Total: {state['total_loss']:.5f}"

                def _ui_update():
                    self.var_progress.set(pct)
                    self.var_progress_text.set(summary)
                    self.loss_chart.push(state['iter'], state['total_loss'], state.get('losses'))
                    if preview is not None:
                        self.paint_canvas.set_live_preview(preview)

                self.after(0, _ui_update)

            pme = PaintMeshEditing(cfg['pme_args'])

            if cfg['multiviews']:
                view_constraints = []
                for mv in cfg['multiviews']:
                    h, w = mv['render_size']
                    label_t = torch.FloatTensor(mv['label_bgr']).to(device) / 255.0 if mv['label_bgr'] is not None else None
                    if mv['mask_gray'] is not None and np.any(mv['mask_gray'] > 0):
                        mask_t = torch.FloatTensor(mv['mask_gray']).to(device).unsqueeze(-1) / 255.0
                    else:
                        mask_t = torch.ones((h, w, 1), device=device)
                    normal_t = torch.FloatTensor(mv['normal_target']).to(device) if mv['normal_target'] is not None else None
                    depth_t = torch.FloatTensor(mv['depth_target']).to(device) if mv['depth_target'] is not None else None
                    sil_t = torch.FloatTensor(mv['silhouette_target']).to(device) if mv['silhouette_target'] is not None else None
                    view_constraints.append(
                        ViewConstraint(
                            intrinsic=mv['intrinsic'],
                            extrinsic=mv['extrinsic'],
                            label_image=label_t,
                            mask_image=mask_t,
                            normal_image=normal_t,
                            depth_image=depth_t,
                            silhouette_image=sil_t,
                            render_size=mv['render_size'],
                            weight=mv['weight'],
                            name=mv['name'],
                        )
                    )

                result = pme.optimize_multiview(
                    mesh=mesh,
                    views=view_constraints,
                    renderer=renderer,
                    out_dir=cfg['out_dir'],
                    callback=on_step,
                    stop_event=self._stop_event,
                )
            else:
                renderer.set_camera_intrinsic(cfg['intrinsic'])
                renderer.set_camera_extrinsic(cfg['extrinsic'])

                label_t = torch.FloatTensor(cfg['label_bgr']).to(device) / 255.0
                if cfg['mask_gray'] is not None and np.any(cfg['mask_gray'] > 0):
                    mask_t = torch.FloatTensor(cfg['mask_gray']).to(device).unsqueeze(-1) / 255.0
                else:
                    mask_t = torch.ones((cfg['render_size'][0], cfg['render_size'][1], 1), device=device)

                normal_t = torch.FloatTensor(cfg['normal_target']).to(device) if cfg['normal_target'] is not None else None
                depth_t = torch.FloatTensor(cfg['depth_target']).to(device) if cfg['depth_target'] is not None else None
                sil_t = torch.FloatTensor(cfg['silhouette_target']).to(device) if cfg['silhouette_target'] is not None else None

                result = pme.optimize(
                    mesh=mesh,
                    label_image=label_t,
                    mask_image=mask_t,
                    renderer=renderer,
                    normal_image=normal_t,
                    depth_image=depth_t,
                    silhouette_image=sil_t,
                    out_dir=cfg['out_dir'],
                    callback=on_step,
                    stop_event=self._stop_event,
                )

            self.current_mesh = mesh
            out_obj = os.path.join(cfg['out_dir'], 'deformed.obj')
            save_obj(out_obj, mesh)

            deformed_vs = mesh.vs.detach().cpu().numpy()
            status_msg = f"Finished {result['completed_iters']} iters ({result.get('num_views', 1)}v). Saved: {out_obj}"
            self.log(status_msg)
            self.after(0, lambda: self.var_progress_text.set(status_msg))
            self.after(0, lambda vs=deformed_vs: self.viewport3d.update_vertices(vs))
        except Exception as exc:
            self.log(f"[Error] Optimization failed: {exc}")
            self.after(0, lambda e=str(exc): messagebox.showerror("Optimization Error", e))
            self.after(0, lambda: self.var_progress_text.set("Failed"))
        finally:
            def _reset_buttons():
                self.btn_start_opt.configure(state='normal')
                self.btn_stop_opt.configure(state='disabled')
            self.after(0, _reset_buttons)


def launch_gui(initial_sample=None):
    app = PaintMeshEditingApp(initial_sample=initial_sample)
    app.mainloop()
