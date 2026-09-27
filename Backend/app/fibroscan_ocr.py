import os
import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#_tesseract_cmd = os.getenv("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
# if _tesseract_cmd:
#     pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def stage_fibrosis(e_kpa: float) -> str:
    """VCTE staging of fibrosis, based on E (kPa)."""
    if e_kpa < 8:
        return "F0-F1 (No/minimal fibrosis)"
    elif 8 <= e_kpa < 12:
        return "F2-F3 (Significant fibrosis)"
    else:  # >= 12
        return "F3-F4 (Advanced fibrosis)"


def stage_steatosis(cap_dbm: float) -> str:
    """CAP staging of steatosis, based on CAP (dB/m)."""
    if cap_dbm < 248:
        return "S0 (Normal)"
    elif 248 <= cap_dbm <= 267:
        return "S1 (Mild)"
    elif 268 <= cap_dbm <= 280:
        return "S2 (Moderate)"
    else:  # >= 280
        return "S3 (Severe)"



CYAN_HSV_LOW = np.array([80, 60, 100])
CYAN_HSV_HIGH = np.array([100, 255, 255])
ORANGE_HSV_LOW = np.array([10, 80, 100])
ORANGE_HSV_HIGH = np.array([30, 255, 255])

_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?%?$")


def _color_mask(img_bgr, low, high):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _ocr_numeric_boxes(mask, debug=False, label=""):
    data = pytesseract.image_to_data(mask, config="--psm 11", output_type=Output.DICT)
    boxes = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        if debug:
            print(f"    [{label}] '{text}' h={data['height'][i]} conf={data['conf'][i]}")
        if not _NUMBER_RE.match(text):
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1.0
        boxes.append({
            "text": text,
            "value": float(text.rstrip("%")),
            "has_pct": text.endswith("%"),
            "height": data["height"][i],
            "x": data["left"][i],
            "y": data["top"][i],
            "conf": conf,
        })
    return boxes


def _split_main_and_small(boxes, min_conf=30):
    good = [b for b in boxes if b["conf"] >= min_conf] or boxes
    if not good:
        return None, None
    good.sort(key=lambda b: b["height"], reverse=True)
    main = good[0]
    secondary = next((b for b in good[1:]), None)
    return main, secondary


def extract_fibroscan_values_v2(image_bytes_or_path, debug: bool = False) -> dict:
    
    if isinstance(image_bytes_or_path, (bytes, bytearray)):
        arr = np.frombuffer(image_bytes_or_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_bytes_or_path)

    if img is None:
        raise ValueError("Could not decode the provided FibroScan image")

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    cyan_mask = _color_mask(img, CYAN_HSV_LOW, CYAN_HSV_HIGH)
    orange_mask = _color_mask(img, ORANGE_HSV_LOW, ORANGE_HSV_HIGH)

    if debug:
        print("Cyan-channel numeric boxes:")
    cyan_boxes = _ocr_numeric_boxes(cyan_mask, debug=debug, label="cyan")
    if debug:
        print("Orange-channel numeric boxes:")
    orange_boxes = _ocr_numeric_boxes(orange_mask, debug=debug, label="orange")

    cap_main, _ = _split_main_and_small(cyan_boxes)
    e_main, _ = _split_main_and_small(orange_boxes)

    cap_value = cap_main["value"] if cap_main else None

    sd_value = None
    if cyan_boxes and cap_main:
        band = max(cap_main["height"] * 2, 150)
        nearby = [b for b in cyan_boxes
                  if b is not cap_main and abs(b["y"] - cap_main["y"]) <= band]
        strong = [b for b in nearby if b["conf"] >= 50]
        pool = strong or nearby
        if pool:
            pool.sort(key=lambda b: (b["conf"], b["height"]), reverse=True)
            sd_value = pool[0]["value"]

    e_value = e_main["value"] if e_main else None
    iqr_value = None
    if orange_boxes and e_main:
        band = max(e_main["height"] * 2, 150)
        nearby = [b for b in orange_boxes
                  if b is not e_main and abs(b["y"] - e_main["y"]) <= band]
        strong = [b for b in nearby if b["conf"] >= 50]
        pool = strong or nearby
        if pool:
            pool.sort(key=lambda b: (b["conf"], b["height"]), reverse=True)
            iqr_value = pool[0]["value"]

    result = {"CAP": cap_value, "E": e_value, "IQR_med_pct": iqr_value, "SD": sd_value}

    if all(v is None for v in result.values()):
        if debug:
            print("Color isolation found nothing - falling back to whole-image OCR.")
        result = _extract_fibroscan_values_fallback(img, debug=debug)

    return result


def _extract_fibroscan_values_fallback(img, debug: bool = False) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config="--psm 11")
    n = len(data["text"])
    words = []
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        words.append({"text": text, "x": data["left"][i], "y": data["top"][i],
                       "w": data["width"][i], "h": data["height"][i]})

    number_re = re.compile(r"^-?\d+(\.\d+)?%?$")

    def numbers_near(label_words, max_dx=250, max_dy=220, below_only=True):
        candidates = []
        for lw in words:
            if lw["text"].upper().strip(":") in label_words:
                for w in words:
                    if not number_re.match(w["text"]):
                        continue
                    dx = w["x"] - lw["x"]
                    dy = w["y"] - lw["y"]
                    if below_only and dy < 0:
                        continue
                    if abs(dx) <= max_dx and abs(dy) <= max_dy:
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        candidates.append((dist, float(w["text"].rstrip("%"))))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    cap_value = numbers_near({"CAP", "MEAN", "EAN"})
    e_value = numbers_near({"MEDIAN", "E"})
    iqr_value = numbers_near({"IQR/MED.", "IQR/MED", "IQR"})
    if iqr_value is None:
        percent_words = [w for w in words if re.match(r"^\d+(\.\d+)?%$", w["text"])]
        if percent_words:
            iqr_value = float(percent_words[0]["text"].rstrip("%"))
    sd_value = numbers_near({"SD"})

    return {"CAP": cap_value, "E": e_value, "IQR_med_pct": iqr_value, "SD": sd_value}


def interpret_fibroscan(
    image_bytes_or_path,
    manual_override: dict | None = None,
    debug: bool = False,
) -> dict:
    
    values = extract_fibroscan_values_v2(image_bytes_or_path, debug=debug)
    if manual_override:
        values.update(manual_override)

    result = {"raw_values": values}

    if values.get("E") is not None:
        result["fibrosis_stage"] = stage_fibrosis(values["E"])
    else:
        result["fibrosis_stage"] = "Could not read E (kPa) value - please enter manually"

    if values.get("CAP") is not None:
        result["steatosis_stage"] = stage_steatosis(values["CAP"])
    else:
        result["steatosis_stage"] = "Could not read CAP value - please enter manually"

    return result
