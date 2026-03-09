"""
Frame Compositor — Video Overlay Engine
==========================================
Composites visual overlays onto camera frames:
  - Transcript bar (bottom, semi-transparent)
  - Sign detection box (top-right, current sign + confidence)
  - Hand-adjacent label (sign + confidence drawn near wrist)
  - Status dot (top-left, pipeline running indicator)

All overlays are optional and controlled by pipeline settings.

Note: Hand landmarks are drawn by the Landmarker (MediaPipe drawing utils),
NOT by the compositor. The compositor only draws UI overlays.

Usage:
    compositor = FrameCompositor()
    
    composited = compositor.render(
        frame=camera_frame,
        transcript="Hello world",
        sign="A",
        confidence=0.95,
        hands_detected=True,
        wrist_position=(0.5, 0.7),
        letter_added=True,
        show_overlay=True,
    )
"""

import cv2
import numpy as np
from typing import Optional


class FrameCompositor:
    """Renders text overlays and status indicators onto video frames."""

    # ── Colors (BGR) ──
    COLOR_BG = (20, 20, 25)
    COLOR_TEXT_WHITE = (255, 255, 255)
    COLOR_TEXT_DIM = (100, 100, 110)

    # ── Layout constants ──
    TRANSCRIPT_BAR_HEIGHT = 48
    SIGN_BOX_WIDTH = 140
    SIGN_BOX_HEIGHT = 72
    SIGN_BOX_MARGIN = 12
    STATUS_DOT_RADIUS = 6
    STATUS_DOT_MARGIN = 16
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.7
    FONT_SCALE_MEDIUM = 0.5
    FONT_SCALE_SMALL = 0.4

    def __init__(self):
        pass

    def render(
        self,
        frame: np.ndarray,
        transcript: str = "",
        sign: str = "",
        confidence: float = 0.0,
        hands_detected: bool = False,
        wrist_position: Optional[tuple] = None,
        letter_added: bool = False,
        show_overlay: bool = True,
        pipeline_running: bool = True,
    ) -> np.ndarray:
        """
        Render all overlays onto a frame.
        
        Args:
            frame: BGR numpy array from camera (landmarks already drawn by Landmarker)
            transcript: Current transcript text
            sign: Currently detected sign/letter
            confidence: Detection confidence (0.0-1.0)
            hands_detected: Whether hands are in frame
            wrist_position: (x, y) normalized wrist coords (0-1), for near-hand label
            letter_added: Whether a letter was just added to the word (changes color)
            show_overlay: Draw transcript bar and sign box
            pipeline_running: Whether the pipeline is active
            
        Returns:
            Composited BGR numpy array (same size as input)
        """
        output = frame.copy()
        h, w = output.shape[:2]

        if show_overlay:
            # Transcript bar at bottom
            if transcript:
                self._draw_transcript_bar(output, transcript, w, h)

            # Sign detection box at top-right
            if sign:
                self._draw_sign_box(output, sign, confidence, w)

            # Sign label near the hand (matches original repo behavior)
            if sign and wrist_position is not None:
                self._draw_hand_label(output, sign, confidence, wrist_position, letter_added, w, h)

        # Status dot (always visible when pipeline is running)
        if pipeline_running:
            self._draw_status_dot(output, hands_detected)

        return output

    # ── Hand-Adjacent Label ─────────────────────

    def _draw_hand_label(
        self,
        frame: np.ndarray,
        sign: str,
        confidence: float,
        wrist_position: tuple,
        letter_added: bool,
        w: int,
        h: int,
    ) -> None:
        """
        Draw the detected sign + confidence near the wrist.
        
        Matches the original sign-language-processing behavior:
            text_x = int(first_landmark[0] * width) - 100
            text_y = int(first_landmark[1] * height) + 50
            cv2.putText(..., color=(0,0,255) if added_letter else (0,255,0), ...)
        """
        text_x = int(wrist_position[0] * w) - 100
        text_y = int(wrist_position[1] * h) + 50

        # Clamp to frame bounds
        text_x = max(10, min(text_x, w - 200))
        text_y = max(30, min(text_y, h - 10))

        label_text = f"{sign} {confidence * 100:.1f}%"

        # Red if letter was added (accepted), Green if just detected (not added)
        color = (0, 0, 255) if letter_added else (0, 255, 0)

        cv2.putText(
            img=frame,
            text=label_text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=4,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    # ── Transcript Bar ──────────────────────────

    def _draw_transcript_bar(
        self, frame: np.ndarray, text: str, w: int, h: int
    ) -> None:
        """Draw a semi-transparent transcript bar at the bottom of the frame."""
        bar_h = self.TRANSCRIPT_BAR_HEIGHT
        bar_y = h - bar_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top border line
        cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 70), 1)

        # Transcript text (truncate if too long)
        max_chars = w // 10
        display_text = text if len(text) <= max_chars else "…" + text[-(max_chars - 1):]

        text_y = bar_y + bar_h // 2 + 6
        cv2.putText(
            frame,
            display_text,
            (16, text_y),
            self.FONT,
            self.FONT_SCALE_MEDIUM,
            self.COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

        # "TRANSCRIPT" label
        cv2.putText(
            frame,
            "TRANSCRIPT",
            (16, bar_y + 14),
            self.FONT,
            0.3,
            self.COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

    # ── Sign Detection Box ──────────────────────

    def _draw_sign_box(
        self, frame: np.ndarray, sign: str, confidence: float, w: int
    ) -> None:
        """Draw the current sign detection in a box at the top-right corner."""
        margin = self.SIGN_BOX_MARGIN
        box_w = self.SIGN_BOX_WIDTH
        box_h = self.SIGN_BOX_HEIGHT
        x1 = w - box_w - margin
        y1 = margin
        x2 = w - margin
        y2 = margin + box_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Border
        border_color = self._confidence_color(confidence)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

        # Sign letter (large, centered)
        text_size = cv2.getTextSize(sign, self.FONT, self.FONT_SCALE_LARGE, 2)[0]
        text_x = x1 + (box_w - text_size[0]) // 2
        text_y = y1 + 32
        cv2.putText(
            frame,
            sign,
            (text_x, text_y),
            self.FONT,
            self.FONT_SCALE_LARGE,
            self.COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )

        # Confidence percentage
        conf_text = f"{confidence * 100:.0f}%"
        conf_size = cv2.getTextSize(conf_text, self.FONT, self.FONT_SCALE_SMALL, 1)[0]
        conf_x = x1 + (box_w - conf_size[0]) // 2
        conf_y = y2 - 10
        cv2.putText(
            frame,
            conf_text,
            (conf_x, conf_y),
            self.FONT,
            self.FONT_SCALE_SMALL,
            self._confidence_color(confidence),
            1,
            cv2.LINE_AA,
        )

        # Confidence bar
        bar_x1 = x1 + 10
        bar_x2 = x2 - 10
        bar_y = y2 - 20
        bar_w = bar_x2 - bar_x1
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 3), (50, 50, 55), -1)
        fill_w = int(bar_w * min(confidence, 1.0))
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x1, bar_y),
                (bar_x1 + fill_w, bar_y + 3),
                self._confidence_color(confidence),
                -1,
            )

    # ── Status Dot ──────────────────────────────

    def _draw_status_dot(
        self, frame: np.ndarray, hands_detected: bool
    ) -> None:
        """Draw a small status dot at the top-left corner."""
        x = self.STATUS_DOT_MARGIN
        y = self.STATUS_DOT_MARGIN
        r = self.STATUS_DOT_RADIUS

        color = (120, 220, 140) if hands_detected else (80, 180, 240)  # Green / Amber
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r + 2, color, 1, cv2.LINE_AA)

        # Label
        label = "HANDS" if hands_detected else "IDLE"
        cv2.putText(
            frame,
            label,
            (x + r + 8, y + 4),
            self.FONT,
            0.35,
            self.COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

    # ── Helpers ─────────────────────────────────

    @staticmethod
    def _confidence_color(confidence: float) -> tuple:
        """Return a BGR color based on confidence level."""
        if confidence >= 0.8:
            return (120, 220, 140)     # Green
        elif confidence >= 0.6:
            return (230, 120, 180)     # Violet
        elif confidence >= 0.4:
            return (80, 180, 240)      # Amber
        else:
            return (80, 80, 230)       # Red
