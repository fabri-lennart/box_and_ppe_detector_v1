import numpy as np
from ultralytics import YOLO

from src import config
from src.models.detection import Detection


class Detector:
	def __init__(
		self,
		model_path: str = str(config.MODEL_PATH),
		confidence: float = config.CONFIDENCE_THRESHOLD,
		iou: float = config.IOU_THRESHOLD,
		classes_of_interest: set[str] | None = None,
	) -> None:
		self.model = YOLO(model_path)
		self.confidence = confidence
		self.iou = iou
		self.class_names = self.model.names
		self.classes_of_interest = classes_of_interest

	def detect(self, frame: np.ndarray) -> list[Detection]:
		results = self.model.predict(
			frame,
			conf=self.confidence,
			iou=self.iou,
			verbose=False,
		)
		detections: list[Detection] = []

		for result in results:
			for box in result.boxes:
				class_id = int(box.cls[0])
				class_name = self.class_names[class_id]
				if self.classes_of_interest and class_name not in self.classes_of_interest:
					continue

				x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
				detections.append(
					Detection(
						class_id=class_id,
						class_name=class_name,
						confidence=float(box.conf[0]),
						x1=x1,
						y1=y1,
						x2=x2,
						y2=y2,
					)
				)

		return detections
