from typing import Self

import cv2
import numpy as np

from src import config


class Camera:
	def __init__(self, index: int = config.CAMERA_INDEX) -> None:
		self.index = index
		self._cap: cv2.VideoCapture | None = None
		self._configure()

	def _configure(self) -> None:
		cap = cv2.VideoCapture(self.index)

		if not cap.isOpened():
			cap.release()
			raise RuntimeError(f"Could not open camera at index {self.index}")

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
		cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)
		self._cap = cap

	def read(self) -> np.ndarray | None:
		if self._cap is None:
			return None

		ok, frame = self._cap.read()
		return frame if ok else None

	def release(self) -> None:
		if self._cap is not None:
			self._cap.release()
			self._cap = None

	def __enter__(self) -> Self:
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		self.release()
