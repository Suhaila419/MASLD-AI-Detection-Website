import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app import config, preprocessing, schemas
from app import fibroscan_ocr
from app.models_loader import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("masld_api")

app = FastAPI(
    title="MASLD Diagnosis API",
    description="Segmentation + Classification (image & tabular) API for liver ultrasound / FibroScan data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.on_event("startup")
def load_models_on_startup():
    registry.load_all()


def _read_upload_image_bytes(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    return file.file.read()


@app.get("/health", response_model=schemas.HealthResponse, tags=["health"])
def health():
    loaded = all(
        m is not None
        for m in [
            registry.unet_model,
            registry.densenet_fibrosis,
            registry.densenet_steatosis,
            registry.densenet_grades,
            registry.xgb_fibrosis,
            registry.xgb_steatosis,
            registry.fibrosis_label_encoders,
            registry.fibrosis_scaler,
            registry.steatosis_label_encoders,
            registry.steatosis_scaler,
        ]
    )
    return schemas.HealthResponse(status="ok" if loaded else "loading", models_loaded=loaded)


@app.post("/segment", response_model=schemas.SegmentationResponse, tags=["segmentation"])
async def segment(image: UploadFile = File(...)):
    """يرجع الـ mask الناتج من موديل الـ Attention U-Net كـ PNG مُرمّز base64."""
    image_bytes = _read_upload_image_bytes(image)

    try:
        _, mask_tensor = preprocessing.segment_image_bytes(image_bytes, registry.unet_model)
    except Exception as exc:
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {exc}")

    mask_np = (mask_tensor.numpy().squeeze() * 255).astype(np.uint8)  # (224,224) 0/255
    pil_mask = Image.fromarray(mask_np, mode="L")

    buffer = io.BytesIO()
    pil_mask.save(buffer, format="PNG")
    mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return schemas.SegmentationResponse(
        mask_base64_png=mask_b64,
        width=config.IMG_WIDTH,
        height=config.IMG_HEIGHT,
    )


@app.post(
    "/extract/fibroscan",
    response_model=schemas.FibroscanExtractResponse,
    tags=["fibroscan-ocr"],
)
async def extract_fibroscan(image: UploadFile = File(...)):
    image_bytes = _read_upload_image_bytes(image)

    try:
        report = fibroscan_ocr.interpret_fibroscan(image_bytes)
    except Exception as exc:
        logger.exception("FibroScan OCR extraction failed")
        raise HTTPException(status_code=400, detail=f"FibroScan extraction failed: {exc}")

    return schemas.FibroscanExtractResponse(
        raw_values=schemas.FibroscanRawValues(**report["raw_values"]),
        fibrosis_stage=report["fibrosis_stage"],
        steatosis_stage=report["steatosis_stage"],
    )


@app.post(
    "/predict/fibrosis/image",
    response_model=schemas.ClassificationResponse,
    tags=["image-classification"],
)
async def predict_fibrosis_image(image: UploadFile = File(...)):
    image_bytes = _read_upload_image_bytes(image)
    try:
        label, confidence, probs = preprocessing.predict_class_from_image(
            image_bytes,
            registry.unet_model,
            registry.densenet_fibrosis,
            config.FIBROSIS_IMAGE_CLASS_NAMES,
        )
    except Exception as exc:
        logger.exception("Fibrosis image prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return schemas.ClassificationResponse(
        predicted_class=label, confidence=confidence, probabilities=probs
    )



@app.post(
    "/predict/steatosis/image",
    response_model=schemas.ClassificationResponse,
    tags=["image-classification"],
)
async def predict_steatosis_image(image: UploadFile = File(...)):
    image_bytes = _read_upload_image_bytes(image)
    try:
        label, confidence, probs = preprocessing.predict_class_from_image(
            image_bytes,
            registry.unet_model,
            registry.densenet_steatosis,
            config.STEATOSIS_IMAGE_CLASS_NAMES,
        )
    except Exception as exc:
        logger.exception("Steatosis image prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return schemas.ClassificationResponse(
        predicted_class=label, confidence=confidence, probabilities=probs
    )



@app.post(
    "/predict/grades/image",
    response_model=schemas.ClassificationResponse,
    tags=["image-classification"],
)
async def predict_grades_image(image: UploadFile = File(...)):
    image_bytes = _read_upload_image_bytes(image)
    try:
        label, confidence, probs = preprocessing.predict_class_from_image(
            image_bytes,
            registry.unet_model,
            registry.densenet_grades,
            config.GRADES_CLASS_NAMES,
        )
    except Exception as exc:
        logger.exception("Grades image prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return schemas.ClassificationResponse(
        predicted_class=label, confidence=confidence, probabilities=probs
    )



@app.post(
    "/predict/fibrosis/tabular",
    response_model=schemas.ClassificationResponse,
    tags=["tabular-classification"],
)
def predict_fibrosis_tabular(payload: schemas.TabularFeatures):
    try:
        X = preprocessing.preprocess_tabular(
            payload.to_raw_dict(),
            registry.fibrosis_label_encoders,
            registry.fibrosis_scaler,
        )
        pred_idx = int(registry.xgb_fibrosis.predict(X)[0])

        probs = None
        if hasattr(registry.xgb_fibrosis, "predict_proba"):
            proba = registry.xgb_fibrosis.predict_proba(X)[0]
            classes = registry.fibrosis_label_encoders[config.FIBROSIS_TARGET_COL].classes_
            probs = {str(classes[i]): float(proba[i]) for i in range(len(classes))}

        label = preprocessing.decode_target_label(
            registry.fibrosis_label_encoders, config.FIBROSIS_TARGET_COL, pred_idx
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Fibrosis tabular prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    confidence = probs[label] if probs else 1.0
    return schemas.ClassificationResponse(
        predicted_class=str(label), confidence=confidence, probabilities=probs
    )


@app.post(
    "/predict/steatosis/tabular",
    response_model=schemas.ClassificationResponse,
    tags=["tabular-classification"],
)
def predict_steatosis_tabular(payload: schemas.TabularFeatures):
    try:
        X = preprocessing.preprocess_tabular(
            payload.to_raw_dict(),
            registry.steatosis_label_encoders,
            registry.steatosis_scaler,
        )
        pred_idx = int(registry.xgb_steatosis.predict(X)[0])

        probs = None
        if hasattr(registry.xgb_steatosis, "predict_proba"):
            proba = registry.xgb_steatosis.predict_proba(X)[0]
            classes = registry.steatosis_label_encoders[config.STEATOSIS_TARGET_COL].classes_
            probs = {str(classes[i]): float(proba[i]) for i in range(len(classes))}

        label = preprocessing.decode_target_label(
            registry.steatosis_label_encoders, config.STEATOSIS_TARGET_COL, pred_idx
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Steatosis tabular prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    confidence = probs[label] if probs else 1.0
    return schemas.ClassificationResponse(
        predicted_class=str(label), confidence=confidence, probabilities=probs
    )
