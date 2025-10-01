import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from utils import load_from_hf

MODEL_REPO = "rexologue/dendrodefect"


class DiseaseDetector:
    """
    ONNX-инференс YOLOv8/YOLOv12-стека с сырым выходом (1 x (4+nc) x N).
    - Препроцесс: letterbox с сохранением соотношения сторон.
    - Постпроцесс: извлечение боксов/оценок/классов, NMS, обратное преобразование в координаты исходного изображения.
    - Возвращает размеченное BGR-изображение исходного размера и список детекций.
    """

    NAMES = [
        "crown damage",
        "fruiting bodies",
        "hollows on trunk",
        "trunk cracks",
        "trunk damage",
        "trunk rots",
    ]

    def __init__(
        self,
        device: str = "cuda",
        models_dir: Path = Path("~/.dendrocache"),
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        imgsz: int | Tuple[int, int] = 640,
        nms_max_detections: int = 300,
        seed: int = 42,
    ) -> None:
        
        self.device = device.lower()
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.nms_max_detections = int(nms_max_detections)

        self._models_dir = models_dir.expanduser().resolve()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        diseaser_dir = self._models_dir / "diseaser"
        diseaser_dir.mkdir(parents=True, exist_ok=True)

        self.onnx_path = diseaser_dir / "model.onnx"

        if not self.onnx_path.exists():
            load_from_hf(MODEL_REPO, "model.onnx", self.onnx_path)

        # фиксированная палитра, чтобы цвета не "скакали" между запусками
        rng = np.random.default_rng(seed)
        self.colors = rng.uniform(0, 255, size=(len(self.NAMES), 3)).astype(np.uint8)

        # создать ONNXRuntime с нужным провайдером
        self.session, (self.input_name, self.in_h, self.in_w) = self._init_session(imgsz)

        # быстрая проверка соответствия числа классов
        # (если модель была обучена на другом количестве классов, просто будут лишние оценки — мы возьмём первые 4+len(NAMES))
        # ничего не меняем, но оставляем комментарий для наглядности
        # print(f"[INFO] Model input: NCHW=1x3x{self.in_h}x{self.in_w}; classes={len(self.NAMES)}")

    # ---------- session / providers ----------

    def _init_session(self, imgsz: int | Tuple[int, int]) -> Tuple[ort.InferenceSession, Tuple[str, int, int]]:
        avail = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

        if self.device.startswith("cuda") and "CUDAExecutionProvider" in avail:
            # Извлечь device_id из 'cuda:5' → 5
            m = re.match(r"cuda(?::(\d+))?$", self.device)
            dev_id = int(m.group(1)) if m and m.group(1) is not None else 0
            providers = [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]
            provider_options = None  # когда указываем опции прямо в providers, provider_options не используется
        else:
            # остаёмся на CPU
            pass

        sess = ort.InferenceSession(self.onnx_path, providers=providers)

        # считаем форму входа: [N, C, H, W]; бывает dynamic → там могут быть str/None
        i0 = sess.get_inputs()[0]
        in_name = i0.name
        shape = list(i0.shape)  # [batch, 3, H, W]
        H, W = shape[2], shape[3]

        def _dim(v, default_):
            return int(v) if isinstance(v, (int, np.integer)) else int(default_)

        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        in_h = _dim(H, imgsz[0])
        in_w = _dim(W, imgsz[1])

        return sess, (in_name, in_h, in_w)

    # ---------- preprocessing (letterbox) ----------

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scaleFill: bool = False,
        scaleup: bool = True,
        stride: int = 32,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Масштабирует изображение с сохранением пропорций и добавляет паддинг до (new_shape).
        Возвращает: изображение BGR, gain, (dw, dh).
        """
        shape = image.shape[:2]  # (h, w)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # вычислить новые размеры и паддинг
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # (w_pad, h_pad)

        if auto:  # кратность stride (не обязательно)
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return image, r, (dw, dh)

    # ---------- postprocessing helpers ----------

    @staticmethod
    def _nms(
        boxes_xywh: List[List[int]],
        scores: List[float],
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ) -> List[int]:
        if not boxes_xywh:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres, top_k=max_det)
        if len(idxs) == 0:
            return []
        return np.array(idxs).flatten().tolist()

    def _draw(self, img: np.ndarray, box_xywh: List[int], cls_id: int, score: float) -> None:
        x, y, w, h = box_xywh
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        color = tuple(int(c) for c in self.colors[cls_id])
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{self.NAMES[cls_id]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(p1[1] - 3, th + 2)
        cv2.rectangle(img, (p1[0], y_text - th - 2), (p1[0] + tw + 2, y_text + 2), color, cv2.FILLED)
        cv2.putText(img, label, (p1[0] + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---------- main API ----------

    def detect(self, image_path: Path, score_threshold: float) -> Tuple[np.ndarray, List[Dict]]:
        """
        :param image_path: путь к изображению любого разрешения
        :return: (annotated_bgr, detections)
                 detections: список словарей {bbox:[x,y,w,h], class_id:int, class_name:str, score:float}
        """
        orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if orig is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        oh, ow = orig.shape[:2]
        # letterbox -> NCHW float32 [0,1]
        lb, gain, (dw, dh) = self._letterbox(orig.copy(), (self.in_h, self.in_w))
        img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1x3xH xW

        # инференс
        out = self.session.run(None, {self.input_name: img})[0]  # ожидаем (1, 4+nc, N) или (1, 5+nc, N)
        out = np.squeeze(out)  # (4+nc, N) или (5+nc, N) # type: ignore
        if out.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
        out = out.transpose(1, 0)  # (N, 4+nc) или (N, 5+nc)

        n_classes = len(self.NAMES)
        dim = out.shape[1]

        # разбор формата: 4(box) + {cls or (obj+cls)}
        has_obj = (dim == 5 + n_classes)
        boxes_cxcywh = out[:, :4]
        if has_obj:
            obj = out[:, 4:5]
            cls_scores = out[:, 5:5 + n_classes]
            scores_per_class = obj * cls_scores
        else:
            cls_scores = out[:, 4:4 + n_classes]
            scores_per_class = cls_scores

        # выбрать лучший класс и оценку для каждой гипотезы
        class_ids = np.argmax(scores_per_class, axis=1)
        scores = scores_per_class[np.arange(scores_per_class.shape[0]), class_ids]

        # отфильтровать по conf_thres
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return orig, []  # нет детекций

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep].astype(float)
        class_ids = class_ids[keep].astype(int)

        # перевод в xyxy в координатах letterbox-карты
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # снять паддинг/масштаб и перевести в координаты исходного изображения
        # обратная трансформация: (x - dw, y - dh) / gain
        x1 = (x1 - dw) / gain
        y1 = (y1 - dh) / gain
        x2 = (x2 - dw) / gain
        y2 = (y2 - dh) / gain

        # в целочисленные xywh, обрезка по границам исходника
        x1 = np.clip(x1, 0, ow - 1)
        y1 = np.clip(y1, 0, oh - 1)
        x2 = np.clip(x2, 0, ow - 1)
        y2 = np.clip(y2, 0, oh - 1)

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).round().astype(int).tolist()
        scores_list = scores.tolist()
        class_ids_list = class_ids.tolist()

        # NMS по исходным координатам
        keep_idx = self._nms(boxes_xywh, scores_list, self.conf_thres, self.iou_thres, self.nms_max_detections)

        detections: List[Dict] = []
        annotated = orig.copy()

        for i in keep_idx:
            score = float(scores_list[i])

            if score < score_threshold:
                continue

            cls_id = class_ids_list[i]
            box = boxes_xywh[i]
            # рисуем
            self._draw(annotated, box, cls_id, score)
            # добавляем в результат
            detections.append(
                {
                    "bbox": box,  # [x, y, w, h] в пикселях исходного изображения
                    "class_id": int(cls_id),
                    "class_name": self.NAMES[cls_id],
                    "score": score,
                }
            )

        return annotated, detections

