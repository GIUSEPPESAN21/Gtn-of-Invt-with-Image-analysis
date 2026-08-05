# -*- coding: utf-8 -*-
"""
Pipeline de visión por computador local (sin IA generativa) para OSIRIS.
Combina clasificación YOLO, OCR y geometría del bounding box en el mismo
esquema de salida que antes producía Gemini, para no romper el flujo de
creación/vinculación de artículos.
"""
import logging
import re

from PIL import Image, ImageOps
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "Electrónica y Cómputo": [
        "laptop", "computer", "keyboard", "mouse", "monitor", "phone", "cellular",
        "camera", "printer", "headset", "earphone", "remote", "modem", "router",
        "speaker", "screen", "hard disc", "disk drive", "projector", "tablet",
    ],
    "Electrodomésticos": [
        "microwave", "refrigerator", "washer", "dishwasher", "toaster", "oven",
        "blender", "espresso", "coffee maker", "iron", "vacuum", "fan", "heater",
    ],
    "Herramientas y Ferretería": [
        "screwdriver", "wrench", "drill", "hammer", "plane", "chain saw", "nail",
        "power drill", "hatchet", "shovel",
    ],
    "Ropa y Accesorios": [
        "jersey", "suit", "shirt", "shoe", "sandal", "boot", "hat", "cap", "sock",
        "tie", "jean", "sweater", "backpack", "bag", "wallet", "sunglass", "glove", "belt",
    ],
    "Muebles y Hogar": [
        "chair", "table", "desk", "sofa", "couch", "lamp", "bed", "cabinet",
        "shelf", "wardrobe", "mirror", "rug",
    ],
    "Alimentos y Bebidas": [
        "banana", "orange", "apple", "bottle", "wine", "beer", "cup", "food",
        "pizza", "bread", "fruit", "vegetable", "can",
    ],
    "Papelería y Oficina": [
        "pencil", "pen", "notebook", "book", "envelope", "binder", "folder",
        "stapler", "scale", "calculator",
    ],
    "Deportes y Recreación": [
        "ball", "racket", "bicycle", "skateboard", "ski", "surfboard", "helmet", "dumbbell",
    ],
    "Juguetes": ["toy", "teddy", "doll", "puzzle", "lego"],
}
DEFAULT_CATEGORY = "Sin categorizar"


def guess_category(label: str, ocr_text: str = "") -> str:
    """Heurística de palabras clave para sugerir una categoría de inventario sin IA generativa."""
    haystack = f"{label} {ocr_text}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category
    return DEFAULT_CATEGORY


class YoloClassifier:
    """Clasificación (no detección) sobre un recorte ya aislado por el detector YOLO."""

    def __init__(self, weights_path: str = "yolov8n-cls.pt"):
        self.model = YOLO(weights_path)

    def classify(self, crop: Image.Image, top_k: int = 3) -> list:
        try:
            results = self.model.predict(crop, verbose=False)
            probs = results[0].probs
            names = results[0].names
            top_idx = probs.top5[:top_k]
            return [
                {"label": names[i].replace("_", " ").title(), "prob": round(float(probs.data[i]), 4)}
                for i in top_idx
            ]
        except Exception as e:
            logger.warning(f"Clasificación YOLO falló: {e}")
            return []


class OCRReader:
    """Extrae texto visible (marca/modelo/empaque) con Tesseract vía pytesseract."""

    def __init__(self, lang: str = "spa+eng"):
        self.lang = lang
        self._binary_available = True

    def read_text(self, crop: Image.Image) -> str:
        if not self._binary_available:
            return ""
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract no está instalado; OCR deshabilitado.")
            self._binary_available = False
            return ""

        try:
            processed = ImageOps.autocontrast(crop.convert("L"))
            text = pytesseract.image_to_string(processed, lang=self.lang, config="--psm 6")
            return re.sub(r"\s+", " ", text).strip()
        except pytesseract.TesseractNotFoundError:
            logger.warning("Binario 'tesseract' no encontrado en el sistema; OCR deshabilitado para esta sesión.")
            self._binary_available = False
            return ""
        except Exception as e:
            logger.warning(f"OCR falló para esta imagen: {e}")
            return ""


class GeometryAnalyzer:
    """Métricas derivadas del bounding box del detector, sin costo de cómputo adicional."""

    @staticmethod
    def compute(box_xyxy, full_image_size: tuple) -> dict:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        img_w, img_h = full_image_size
        return {
            "ancho_px": int(round(w)),
            "alto_px": int(round(h)),
            "area_px": int(round(w * h)),
            "aspect_ratio": round(w / h, 3),
            "area_relativa_pct": round((w * h) / max(img_w * img_h, 1) * 100, 2),
        }


def analyze_object_cv(crop: Image.Image, class_name_yolo_det: str, confianza_det: float,
                       box_xyxy, full_image_size: tuple,
                       classifier: YoloClassifier, ocr: OCRReader) -> dict:
    """Orquesta clasificación + OCR + geometría en el esquema unificado (compatible con Gemini)."""
    top_preds = classifier.classify(crop, top_k=3)
    ocr_text = ocr.read_text(crop)
    geometry = GeometryAnalyzer.compute(box_xyxy, full_image_size)
    best_label = top_preds[0]["label"] if top_preds else class_name_yolo_det.title()

    caracteristicas_partes = [f"Detección: '{class_name_yolo_det}' ({confianza_det * 100:.0f}% confianza)"]
    if top_preds:
        caracteristicas_partes.append(f"Clasificación: '{best_label}' ({top_preds[0]['prob'] * 100:.1f}%)")
    caracteristicas_partes.append(f"Proporción {geometry['aspect_ratio']}, {geometry['area_relativa_pct']}% del encuadre")
    if ocr_text:
        caracteristicas_partes.append(f"Texto detectado: \"{ocr_text[:80]}\"")

    return {
        "elemento_identificado": best_label,
        "cantidad_aproximada": 1,
        "estado_condicion": "No determinado (revisar visualmente antes de guardar)",
        "caracteristicas_distintivas": " | ".join(caracteristicas_partes),
        "posible_categoria_de_inventario": guess_category(best_label, ocr_text),
        "marca_modelo_sugerido": ocr_text if ocr_text else "No visible",
        "fuente_analisis": "cv_local",
        "clasificacion_top3": top_preds,
        "texto_ocr_detectado": ocr_text,
        "geometria": geometry,
        "confianza_deteccion_yolo": round(float(confianza_det), 3),
    }


def merge_with_gemini(cv_result: dict, gemini_result: dict) -> dict:
    """Fusiona un enriquecimiento opcional de Gemini sin perder las señales de CV local."""
    merged = dict(gemini_result)
    merged["fuente_analisis"] = "hibrido"
    for key in ("clasificacion_top3", "texto_ocr_detectado", "geometria", "confianza_deteccion_yolo"):
        merged[key] = cv_result.get(key)
    return merged


class VisionServices:
    """Punto único de inicialización cacheable para los modelos de visión local."""

    def __init__(self, classifier_weights: str = "yolov8n-cls.pt", ocr_lang: str = "spa+eng"):
        self.classifier = YoloClassifier(classifier_weights)
        self.ocr = OCRReader(ocr_lang)
