import pygame
import librosa
import numpy as np
import os
import traceback
import subprocess
import shutil
import tempfile
import threading

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QColorDialog, QFontDialog, QSpinBox, QComboBox, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QColor, QFont, QPixmap, QImage, QPainter

# --- Configuration & Style (defaults) ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 30

COLOR_BACKGROUND = (10, 10, 20)
COLOR_FOREGROUND = (0, 255, 100)
COLOR_ACCENT = (255, 176, 0)
COLOR_TEXT = (0, 255, 100)
COLOR_VIDEO_BORDER = (0, 150, 50)
COLOR_BOX_BG = (20, 20, 40)
DEFAULT_FONT = "Consolas"

# --- Global Variables ---
audio_data = None
audio_sr = None
audio_duration = 0
current_playback_time = 0.0

main_surface = None
title_font = None
info_font = None
small_font = None
song_title_global = "Audio Visualizer"
audio_path_global = None
ffmpeg_path_global = "ffmpeg"
CHUNK_DURATION_SECONDS = 60

# --- PySide6 UI ---
class VisualizerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer Exporter")
        self.setMinimumWidth(950)
        self.setMinimumHeight(600)
        self.setStyleSheet("QWidget { font-size: 14px; }")

        # Configurable options
        self.audio_path = ""
        self.song_title = "Audio Visualizer"
        self.chunk_duration = 60
        self.ffmpeg_path = "ffmpeg"
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.fps = FPS
        self.color_background = QColor(*COLOR_BACKGROUND)
        self.color_foreground = QColor(*COLOR_FOREGROUND)
        self.color_accent = QColor(*COLOR_ACCENT)
        self.color_text = QColor(*COLOR_TEXT)
        self.color_border = QColor(*COLOR_VIDEO_BORDER)
        self.color_box_bg = QColor(*COLOR_BOX_BG)
        self.font_family = DEFAULT_FONT

        # Widgets
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setAlignment(Qt.AlignTop)

        # Audio file
        h_audio = QHBoxLayout()
        h_audio.addWidget(QLabel("Audio File:"))
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Select audio file...")
        h_audio.addWidget(self.audio_edit)
        btn_audio = QPushButton("Browse")
        btn_audio.setToolTip("Browse for an audio file")
        btn_audio.clicked.connect(self.browse_audio)
        h_audio.addWidget(btn_audio)
        left_layout.addLayout(h_audio)

        # Song title
        h_title = QHBoxLayout()
        h_title.addWidget(QLabel("Song Title:"))
        self.title_edit = QLineEdit(self.song_title)
        self.title_edit.setPlaceholderText("Enter song title...")
        h_title.addWidget(self.title_edit)
        left_layout.addLayout(h_title)

        # Chunk duration
        h_chunk = QHBoxLayout()
        h_chunk.addWidget(QLabel("Chunk Duration (s):"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 3600)
        self.chunk_spin.setValue(self.chunk_duration)
        h_chunk.addWidget(self.chunk_spin)
        left_layout.addLayout(h_chunk)

        # FFmpeg path
        h_ffmpeg = QHBoxLayout()
        h_ffmpeg.addWidget(QLabel("FFmpeg Path:"))
        self.ffmpeg_edit = QLineEdit(self.ffmpeg_path)
        self.ffmpeg_edit.setPlaceholderText("Path to ffmpeg executable")
        h_ffmpeg.addWidget(self.ffmpeg_edit)
        btn_ffmpeg = QPushButton("Browse")
        btn_ffmpeg.setToolTip("Browse for FFmpeg executable")
        btn_ffmpeg.clicked.connect(self.browse_ffmpeg)
        h_ffmpeg.addWidget(btn_ffmpeg)
        left_layout.addLayout(h_ffmpeg)

        # Resolution
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Resolution:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 4096)
        self.width_spin.setValue(self.screen_width)
        self.width_spin.setSuffix(" px")
        h_res.addWidget(self.width_spin)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(self.screen_height)
        self.height_spin.setSuffix(" px")
        h_res.addWidget(self.height_spin)
        left_layout.addLayout(h_res)

        # FPS
        h_fps = QHBoxLayout()
        h_fps.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(self.fps)
        h_fps.addWidget(self.fps_spin)
        left_layout.addLayout(h_fps)

        # Font (dropdown for output font only)
        h_font = QHBoxLayout()
        h_font.addWidget(QLabel("Font:"))
        from PySide6.QtGui import QFontDatabase
        self.font_combo = QComboBox()
        self.font_combo.setEditable(False)
        self.font_db = QFontDatabase()
        font_families = self.font_db.families()
        self.font_combo.addItems(sorted(font_families))
        # Set default font selection
        if self.font_family in font_families:
            self.font_combo.setCurrentText(self.font_family)
        else:
            self.font_combo.setCurrentIndex(0)
            self.font_family = self.font_combo.currentText()
        h_font.addWidget(self.font_combo)
        left_layout.addLayout(h_font)

        # Colors
        h_colors = QHBoxLayout()
        self.color_btns = []
        for label, attr in [
            ("Background", "color_background"),
            ("Foreground", "color_foreground"),
            ("Accent", "color_accent"),
            ("Text", "color_text"),
            ("Border", "color_border"),
            ("Box BG", "color_box_bg"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(f"Set {label.lower()} color")
            btn.clicked.connect(lambda _, a=attr: self.choose_color(a))
            h_colors.addWidget(btn)
            self.color_btns.append(btn)
        left_layout.addLayout(h_colors)

        # Run button
        self.run_btn = QPushButton("Run Visualizer")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 16px; padding: 8px;")
        self.run_btn.clicked.connect(self.on_run)
        left_layout.addWidget(self.run_btn)

        left_layout.addStretch(1)

        # --- Right panel: Example mockup ---
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        self.example_label = QLabel("Example Output Preview")
        self.example_label.setAlignment(Qt.AlignCenter)
        self.example_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        right_layout.addWidget(self.example_label)
        self.example_frame = QLabel()
        self.example_frame.setFixedSize(400, 225)
        self.example_frame.setFrameShape(QFrame.Box)
        self.example_frame.setLineWidth(2)
        self.example_frame.setStyleSheet("background: #222;")
        right_layout.addWidget(self.example_frame)
        right_layout.addStretch(1)

        # Combine left and right panels
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

        # Connect all option changes to update the preview
        self.audio_edit.textChanged.connect(self.update_example)
        self.title_edit.textChanged.connect(self.update_example)
        self.chunk_spin.valueChanged.connect(self.update_example)
        self.ffmpeg_edit.textChanged.connect(self.update_example)
        self.width_spin.valueChanged.connect(self.update_example)
        self.height_spin.valueChanged.connect(self.update_example)
        self.fps_spin.valueChanged.connect(self.update_example)
        self.font_combo.currentTextChanged.connect(self.on_font_change)
        for btn in self.color_btns:
            btn.clicked.connect(self.update_example)
        pygame.font.init()
        self.update_example()

    def browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.ogg *.flac)")
        if path:
            self.audio_edit.setText(path)

    def browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select FFmpeg Executable", "", "ffmpeg*")
        if path:
            self.ffmpeg_edit.setText(path)

    def choose_color(self, attr):
        current = getattr(self, attr)
        color = QColorDialog.getColor(current, self, "Select Color")
        if color.isValid():
            setattr(self, attr, color)
            self.update_example()

    def on_font_change(self, font_name):
        self.font_family = font_name
        # Only update the preview, not the UI font
        self.update_example()

    def on_run(self):
        # Validate required fields
        if not self.audio_edit.text().strip():
            self.audio_edit.setStyleSheet("border: 2px solid red;")
            self.audio_edit.setFocus()
            return
        else:
            self.audio_edit.setStyleSheet("")
        if not self.ffmpeg_edit.text().strip():
            self.ffmpeg_edit.setStyleSheet("border: 2px solid red;")
            self.ffmpeg_edit.setFocus()
            return
        else:
            self.ffmpeg_edit.setStyleSheet("")
        self.run_btn.setEnabled(False)
        # Set globals from UI
        global audio_path_global, song_title_global, CHUNK_DURATION_SECONDS, ffmpeg_path_global
        global SCREEN_WIDTH, SCREEN_HEIGHT, FPS
        global COLOR_BACKGROUND, COLOR_FOREGROUND, COLOR_ACCENT, COLOR_TEXT, COLOR_VIDEO_BORDER, COLOR_BOX_BG, DEFAULT_FONT

        audio_path_global = self.audio_edit.text()
        song_title_global = self.title_edit.text() or "Audio Visualizer"
        CHUNK_DURATION_SECONDS = self.chunk_spin.value()
        ffmpeg_path_global = self.ffmpeg_edit.text() or "ffmpeg"
        SCREEN_WIDTH = self.width_spin.value()
        SCREEN_HEIGHT = self.height_spin.value()
        FPS = self.fps_spin.value()
        COLOR_BACKGROUND = self.color_background.getRgb()[:3]
        COLOR_FOREGROUND = self.color_foreground.getRgb()[:3]
        COLOR_ACCENT = self.color_accent.getRgb()[:3]
        COLOR_TEXT = self.color_text.getRgb()[:3]
        COLOR_VIDEO_BORDER = self.color_border.getRgb()[:3]
        COLOR_BOX_BG = self.color_box_bg.getRgb()[:3]
        DEFAULT_FONT = self.font_family

        threading.Thread(target=main, daemon=True).start()

    def enable_run(self):
        self.run_btn.setEnabled(True)

    def update_example(self):
        if not pygame.font.get_init():
            pygame.font.init()
        width = self.width_spin.value()
        height = self.height_spin.value()
        preview_w, preview_h = 400, 225
        surf = pygame.Surface((width, height))
        try:
            font = pygame.font.SysFont(self.font_family, 48, bold=True)
        except Exception:
            font = pygame.font.Font(None, 48)
        bg = self.color_background.getRgb()[:3]
        surf.fill(bg)
        video_box_rect = pygame.Rect(50, 70, width - 100, height // 3)
        progress_bar_y = video_box_rect.bottom + 20
        left_box_rect = pygame.Rect(50, progress_bar_y + 50, width // 3 - 30, height - (progress_bar_y + 70))
        right_box_rect = pygame.Rect(width // 3 + 30, progress_bar_y + 50, width * 2 // 3 - 80, height - (progress_bar_y + 70))
        surf.fill(self.color_box_bg.getRgb()[:3], video_box_rect)
        surf.fill(self.color_box_bg.getRgb()[:3], left_box_rect)
        surf.fill(self.color_box_bg.getRgb()[:3], right_box_rect)
        pygame.draw.rect(surf, self.color_border.getRgb()[:3], video_box_rect, 3)
        pygame.draw.rect(surf, self.color_border.getRgb()[:3], left_box_rect, 2)
        pygame.draw.rect(surf, self.color_border.getRgb()[:3], right_box_rect, 2)
        draw_text(self.title_edit.text() or "Audio Visualizer", font, self.color_text.getRgb()[:3], surf, width // 2, 30, center=True)
        draw_3d_typography_mountain(
            surf,
            np.linspace(0.2, 0.8, 64),
            left_box_rect.x, left_box_rect.y,
            left_box_rect.width, left_box_rect.height,
            frame_idx=20
        )
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2)).copy()
        img = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.example_frame.setPixmap(pix)

# --- Pygame Initialization ---
def init_pygame_headless():
    global main_surface, title_font, info_font, small_font
    pygame.init()
    main_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    try:
        title_font = pygame.font.SysFont(DEFAULT_FONT, 48, bold=True)
        info_font = pygame.font.SysFont(DEFAULT_FONT, 24)
        small_font = pygame.font.SysFont(DEFAULT_FONT, 18)
    except Exception as e:
        print(f"{DEFAULT_FONT} font not found, using default. Error: {e}")
        title_font = pygame.font.Font(None, 60)
        info_font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 24)
    print("Pygame initialized in headless mode.")

# --- Audio ---
def load_audio_file():
    global audio_data, audio_sr, audio_duration
    try:
        audio_data, audio_sr = librosa.load(audio_path_global, sr=None)
        audio_duration = librosa.get_duration(y=audio_data, sr=audio_sr)
        print(f"Audio loaded: {audio_path_global}, Duration: {audio_duration:.2f}s, SR: {audio_sr}Hz")
        return True
    except Exception as e:
        print(f"Error loading audio file: {e}")
        traceback.print_exc()
        return False

def get_audio_features(time_sec):
    if audio_data is None or audio_sr is None:
        return {
            "amplitude_bars": np.zeros(32),
            "left_vol": 0,
            "right_vol": 0,
            "waveform": np.zeros(200),
            "spectrum": np.zeros(64)
        }
    start_sample = int(time_sec * audio_sr)
    chunk_size = audio_sr // 10
    end_sample = min(start_sample + chunk_size, len(audio_data))
    if start_sample >= end_sample:
        return {
            "amplitude_bars": np.zeros(32),
            "left_vol": 0,
            "right_vol": 0,
            "waveform": np.zeros(200),
            "spectrum": np.zeros(64)
        }
    current_chunk = audio_data[start_sample:end_sample]
    if len(current_chunk) == 0:
        return {
            "amplitude_bars": np.zeros(32),
            "left_vol": 0,
            "right_vol": 0,
            "waveform": np.zeros(200),
            "spectrum": np.zeros(64)
        }
    num_bars = 32
    bar_values = np.zeros(num_bars)
    sub_chunk_size = len(current_chunk) // num_bars
    if sub_chunk_size > 0:
        for i in range(num_bars):
            sub_chunk = current_chunk[i*sub_chunk_size:(i+1)*sub_chunk_size]
            if len(sub_chunk) > 0:
                rms = np.sqrt(np.mean(sub_chunk**2))
                bar_values[i] = np.clip(rms * 5, 0, 1)
    mean_amp = np.mean(np.abs(current_chunk))
    left_vol = np.clip(mean_amp * 10, 0, 1)
    right_vol = np.clip(mean_amp * 10, 0, 1)
    num_wave_points = 200
    waveform_data = np.zeros(num_wave_points)
    step = max(1, len(current_chunk) // num_wave_points)
    waveform_data_raw = current_chunk[::step][:num_wave_points]
    if len(waveform_data_raw) > 0:
        max_val = np.max(np.abs(waveform_data_raw))
        if max_val > 0:
            waveform_data[:len(waveform_data_raw)] = waveform_data_raw / max_val
        else:
            waveform_data[:len(waveform_data_raw)] = waveform_data_raw
    num_spectrum_bins = 64
    spectrum_data = np.zeros(num_spectrum_bins)
    n_fft = 2048
    if len(current_chunk) >= n_fft // 4:
        padded_chunk = librosa.util.fix_length(current_chunk, size=n_fft)
        fft_result = np.abs(librosa.stft(padded_chunk, n_fft=n_fft, hop_length=n_fft + 1))
        if fft_result.size > 0:
            fft_result_db = librosa.amplitude_to_db(fft_result[:,0], ref=np.max)
            relevant_fft = fft_result_db[:n_fft // 2]
            bin_size = len(relevant_fft) // num_spectrum_bins
            if bin_size > 0:
                for i in range(num_spectrum_bins):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, len(relevant_fft))
                    if start_idx < end_idx:
                        spectrum_data[i] = np.mean(relevant_fft[start_idx:end_idx])
                spectrum_data = (np.clip(spectrum_data, -60, 0) + 60) / 60
    return {
        "amplitude_bars": bar_values,
        "left_vol": left_vol,
        "right_vol": right_vol,
        "waveform": waveform_data,
        "spectrum": spectrum_data
    }

# --- Drawing Functions ---
def draw_text(text, font, color, surface, x, y, center=False):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    if center:
        textrect.center = (x, y)
    else:
        textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def draw_song_title(surface, title):
    draw_text(title, title_font, COLOR_TEXT, surface, SCREEN_WIDTH // 2, 30, center=True)

def draw_progress_bar(surface, current_time, total_duration):
    bar_y = 70 + (SCREEN_HEIGHT // 3) + 20
    bar_width = SCREEN_WIDTH - 100
    bar_height = 20
    progress_rect_bg = pygame.Rect(50, bar_y, bar_width, bar_height)
    pygame.draw.rect(surface, COLOR_VIDEO_BORDER, progress_rect_bg, 2)
    if total_duration > 0:
        progress_ratio = min(current_time / total_duration, 1.0)
        filled_width = int(progress_ratio * bar_width)
        progress_rect_fg = pygame.Rect(50, bar_y, filled_width, bar_height)
        pygame.draw.rect(surface, COLOR_ACCENT, progress_rect_fg)
    time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d} / {int(total_duration // 60):02d}:{int(total_duration % 60):02d}"
    draw_text(time_str, small_font, COLOR_TEXT, surface, SCREEN_WIDTH // 2, bar_y + bar_height + 15, center=True)

# --- 3D Typography Mountain Grid (Bottom Left) ---
def draw_3d_typography_mountain(surface, spectrum, x, y, area_width, area_height, frame_idx):
    grid_x = 5
    grid_y = 5
    margin = 20
    # Plane and camera setup
    cx = x + area_width // 2
    cy = y + area_height // 2
    plane_w = area_width - 2 * margin
    plane_h = area_height - 2 * margin
    max_z = 0.18 * min(area_width, area_height)
    # Rotation: lay flat (X axis 90deg), rotate Y continually
    angle_y = (frame_idx * 1.2) % 360
    angle_x = 90  # lay flat
    rad_y = np.deg2rad(angle_y)
    rad_x = np.deg2rad(angle_x)
    # Build 3D grid
    points_3d = []
    for gx in range(grid_x):
        for gy in range(grid_y):
            u = gx / (grid_x - 1)
            v = gy / (grid_y - 1)
            px = (u - 0.5) * plane_w
            py = (v - 0.5) * plane_h
            # Use a blend of spectrum bins and a 2D wave for all points
            spec_idx = int((u + v) / 2 * (len(spectrum) - 1))
            spec_val = spectrum[spec_idx]
            t = frame_idx * 0.08
            wave = np.sin(2 * np.pi * (u + t)) * np.cos(2 * np.pi * (v + t))
            pz = (spec_val * 0.7 + 0.3 * wave) * max_z
            # Rotate around X (lay flat) then Y (spin)
            py2 = py * np.cos(rad_x) - pz * np.sin(rad_x)
            pz2 = py * np.sin(rad_x) + pz * np.cos(rad_x)
            px3 = px * np.cos(rad_y) + pz2 * np.sin(rad_y)
            pz3 = -px * np.sin(rad_y) + pz2 * np.cos(rad_y)
            points_3d.append((px3, py2, pz3))
    # Project to 2D (simple perspective)
    projected = []
    cam_z = 900
    for px, py, pz in points_3d:
        zz = cam_z + pz
        sx = int(cx + px * cam_z / zz)
        sy = int(cy - py * cam_z / zz)
        sx = max(x + margin, min(x + area_width - margin, sx))
        sy = max(y + margin, min(y + area_height - margin, sy))
        projected.append((sx, sy))
    # Draw grid lines (horizontal and vertical)
    for gx in range(grid_x):
        for gy in range(grid_y - 1):
            idx1 = gx * grid_y + gy
            idx2 = gx * grid_y + (gy + 1)
            pygame.draw.line(surface, COLOR_ACCENT, projected[idx1], projected[idx2], 2)
    for gy in range(grid_y):
        for gx in range(grid_x - 1):
            idx1 = gx * grid_y + gy
            idx2 = (gx + 1) * grid_y + gy
            pygame.draw.line(surface, COLOR_FOREGROUND, projected[idx1], projected[idx2], 2)
    # Draw vertical bars for each grid point (subtle, not shooting out)
    for gx in range(grid_x):
        for gy in range(grid_y):
            idx = gx * grid_y + gy
            sx, sy = projected[idx]
            bar_len = 8 + int(points_3d[idx][2] * 0.04)
            pygame.draw.line(surface, COLOR_ACCENT, (sx, sy), (sx, sy - bar_len), 2)
    # Draw 3D "AUDIO" text floating above the plane
    try:
        font3d = title_font
        if font3d is None:
            # fallback for preview/mockup
            font3d = pygame.font.Font(None, 48)
        text = "AUDIO"
        text_surface = font3d.render(text, True, COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(cx, y + margin + 10))
        surface.blit(text_surface, text_rect)
    except Exception as e:
        # If font3d is still None or rendering fails, skip drawing text in preview
        pass

def draw_stereo_volume_bars(surface, left_vol, right_vol, x, y, area_width, area_height):
    bar_max_h = area_height - 10
    bar_w = area_width // 2 - 10
    left_h = int(np.clip(left_vol, 0, 1) * bar_max_h)
    pygame.draw.rect(surface, COLOR_FOREGROUND, (x, y + (bar_max_h - left_h), bar_w, left_h))
    draw_text("L", small_font, COLOR_TEXT, surface, x + bar_w // 2, y + bar_max_h + 5, center=True)
    right_h = int(np.clip(right_vol, 0, 1) * bar_max_h)
    pygame.draw.rect(surface, COLOR_FOREGROUND, (x + bar_w + 20, y + (bar_max_h - right_h), bar_w, right_h))
    draw_text("R", small_font, COLOR_TEXT, surface, x + bar_w + 20 + bar_w // 2, y + bar_max_h + 5, center=True)

def draw_audio_wave(surface, data, x, y, area_width, area_height):
    num_points = len(data)
    if num_points < 2: return
    half_height = area_height // 2
    points = []
    for i, val in enumerate(data):
        px = x + (i / (num_points -1)) * area_width if num_points > 1 else x
        py = y + half_height - int(np.clip(val, -1, 1) * (half_height - 5))
        points.append((px, py))
    if len(points) >= 2:
        pygame.draw.lines(surface, COLOR_ACCENT, False, points, 2)

def draw_spectral_waves(surface, data, x, y, area_width, area_height):
    num_bands = len(data)
    if num_bands == 0: return
    band_spacing = 1
    band_width = (area_width - (num_bands - 1) * band_spacing) / num_bands
    if band_width <=0: band_width = 1
    max_band_h = area_height - 10
    for i, val in enumerate(data):
        band_h = int(np.clip(val, 0, 1) * max_band_h)
        band_x = x + i * (band_width + band_spacing)
        band_y_pos = y + (max_band_h - band_h)
        pygame.draw.rect(surface, COLOR_ACCENT, (band_x, band_y_pos, band_width, band_h))

# --- Chunk Export Function ---
def export_video_chunk(frames_list, chunk_index, chunk_start_time, chunk_actual_duration, full_audio_path):
    if not frames_list:
        print(f"Chunk {chunk_index}: No frames to export.")
        return
    output_filename = f"audio_visualizer_output_chunk_{chunk_index}.mp4"
    print(f"Exporting chunk {chunk_index} ({len(frames_list)} frames, audio {chunk_start_time:.2f}s - {chunk_start_time+chunk_actual_duration:.2f}s) to {output_filename}...")
    temp_dir = tempfile.mkdtemp(prefix=f"chunk_{chunk_index}_frames_")
    try:
        for idx, frame in enumerate(frames_list):
            frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((1080, 1920), Image.LANCZOS)
            img.save(frame_path)
        images_pattern = os.path.join(temp_dir, "frame_%05d.png")
        audio_args = []
        if full_audio_path and os.path.exists(full_audio_path):
            audio_args = [
                "-ss", str(chunk_start_time),
                "-t", str(chunk_actual_duration),
                "-i", full_audio_path
            ]
        ffmpeg_cmd = [
            ffmpeg_path_global,
            "-y",
            "-framerate", str(FPS),
            "-i", images_pattern,
        ]
        if audio_args:
            ffmpeg_cmd += audio_args
        ffmpeg_cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
        ]
        if audio_args:
            ffmpeg_cmd += [
                "-c:a", "aac",
                "-shortest"
            ]
        ffmpeg_cmd.append(output_filename)
        print("Running FFmpeg command:", " ".join(ffmpeg_cmd))
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg error:", result.stderr)
        else:
            print(f"Chunk {chunk_index} exported successfully to {output_filename}")
    except Exception as e:
        print(f"Error during video export for chunk {chunk_index}: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- Main Application Logic ---
def main():
    global current_playback_time, main_surface
    init_pygame_headless()
    if not load_audio_file():
        print("Failed to load audio file. Exiting.")
        pygame.quit()
        # Use enable_run() instead of config(state=NORMAL)
        run_button.enable_run()
        return
    current_playback_time = 0.0
    frame_count_for_simulated_time = 0
    current_chunk_frames = []
    chunk_number = 0
    current_chunk_start_time_audio = 0.0
    total_frames_to_process = int(audio_duration * FPS)
    processed_frames_count = 0
    print(f"\nStarting processing. Total audio duration: {audio_duration:.2f}s. Target FPS: {FPS}.")
    print(f"Chunks will be approx {CHUNK_DURATION_SECONDS}s long.")

    # --- Layout box definitions ---
    video_box_rect = pygame.Rect(50, 70, SCREEN_WIDTH - 100, SCREEN_HEIGHT // 3)
    progress_bar_y = video_box_rect.bottom + 20
    left_box_rect = pygame.Rect(50, progress_bar_y + 50, SCREEN_WIDTH // 3 - 30, SCREEN_HEIGHT - (progress_bar_y + 70))
    right_box_rect = pygame.Rect(SCREEN_WIDTH // 3 + 30, progress_bar_y + 50, SCREEN_WIDTH * 2 // 3 - 80, SCREEN_HEIGHT - (progress_bar_y + 70))

    while current_playback_time < audio_duration:
        current_playback_time = frame_count_for_simulated_time / FPS
        if current_playback_time >= audio_duration:
            break
        audio_features = get_audio_features(current_playback_time)
        main_surface.fill(COLOR_BACKGROUND)

        # --- Top blank video area box ---
        pygame.draw.rect(main_surface, COLOR_BOX_BG, video_box_rect)
        pygame.draw.rect(main_surface, COLOR_VIDEO_BORDER, video_box_rect, 3)

        draw_song_title(main_surface, song_title_global)
        draw_progress_bar(main_surface, current_playback_time, audio_duration)

        # --- Left box: 3D Typography Mountain Grid ---
        pygame.draw.rect(main_surface, COLOR_BOX_BG, left_box_rect)
        pygame.draw.rect(main_surface, COLOR_VIDEO_BORDER, left_box_rect, 2)
        draw_3d_typography_mountain(
            main_surface,
            audio_features["spectrum"],
            left_box_rect.x, left_box_rect.y,
            left_box_rect.width, left_box_rect.height,
            frame_count_for_simulated_time
        )

        # --- Right box: Stereo bars, waveform, spectrum ---
        pygame.draw.rect(main_surface, COLOR_BOX_BG, right_box_rect)
        pygame.draw.rect(main_surface, COLOR_VIDEO_BORDER, right_box_rect, 2)
        # Stereo bars
        draw_stereo_volume_bars(
            main_surface,
            audio_features["left_vol"], audio_features["right_vol"],
            right_box_rect.x + 30, right_box_rect.y + 30,
            right_box_rect.width - 60, 80
        )
        # Waveform
        draw_audio_wave(
            main_surface,
            audio_features["waveform"],
            right_box_rect.x + 30, right_box_rect.y + 140,
            right_box_rect.width - 60, 100
        )
        # Spectrum bars
        draw_spectral_waves(
            main_surface,
            audio_features["spectrum"],
            right_box_rect.x + 30, right_box_rect.y + 270,
            right_box_rect.width - 60, right_box_rect.height - 320
        )

        # --- Frame Capturing for Current Chunk ---
        try:
            frame_image_data = pygame.surfarray.array3d(main_surface)
            current_chunk_frames.append(np.transpose(frame_image_data, (1, 0, 2)))
        except Exception as e:
            print(f"Error capturing frame: {e}")
            pass
        frame_count_for_simulated_time += 1
        processed_frames_count += 1
        if processed_frames_count % (FPS * 10) == 0:
            print(f"Processed: {current_playback_time:.2f}s / {audio_duration:.2f}s ({ (current_playback_time/audio_duration)*100 :.1f}%)")
        current_chunk_elapsed_time = current_playback_time - current_chunk_start_time_audio
        if current_chunk_elapsed_time >= CHUNK_DURATION_SECONDS or current_playback_time >= audio_duration:
            actual_chunk_duration = current_chunk_elapsed_time
            if current_playback_time >= audio_duration:
                actual_chunk_duration = audio_duration - current_chunk_start_time_audio
            export_video_chunk(current_chunk_frames, chunk_number, current_chunk_start_time_audio, actual_chunk_duration, audio_path_global)
            current_chunk_frames = []
            chunk_number += 1
            current_chunk_start_time_audio = current_playback_time
            if current_playback_time >= audio_duration:
                break
    if current_chunk_frames:
        print("Exporting final remaining frames...")
        final_chunk_duration = audio_duration - current_chunk_start_time_audio
        export_video_chunk(current_chunk_frames, chunk_number, current_chunk_start_time_audio, final_chunk_duration, audio_path_global)
    pygame.quit()
    print("\nVisualizer processing finished.")
    print(f"A total of {chunk_number + (1 if current_chunk_frames else 0)} chunk(s) should have been generated.")
    # Use enable_run() instead of config(state=NORMAL)
    run_button.enable_run()

# Replace the Tkinter root/mainloop with PySide6
if __name__ == "__main__":
    app = QApplication([])
    ui = VisualizerUI()
    ui.show()
    # Patch run_button.config(state=NORMAL) to ui.enable_run()
    global run_button
    run_button = ui
    app.exec()