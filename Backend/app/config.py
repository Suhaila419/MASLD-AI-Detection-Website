import os

MODELS_DIR = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models", os.path.join(os.path.dirname(__file__), "models"))


def _p(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)



UNET_MODEL_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\AttentionCustomUNet.keras", _p("AttentionCustomUNet.keras"))

DENSENET_FIBROSIS_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\final_DenseNet121_fibrosis.h5", _p("final_DenseNet121_fibrosis.h5"))
DENSENET_STEATOSIS_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\final_DenseNet121(stage_steatosis).h5", _p("final_DenseNet121(stage_steatosis).h5"))
DENSENET_GRADES_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\final_DenseNet121Grades.h5", _p("final_DenseNet121Grades.h5"))

XGB_FIBROSIS_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\XGboost_fibrosis.pickle", _p("XGboost_fibrosis.pickle"))
XGB_STEATOSIS_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\XGboost(stage_steatosis).pickle", _p("XGboost(stage_steatosis).pickle"))

FIBROSIS_LABEL_ENCODERS_PATH = os.getenv(
    "C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\fibrosis_label_encoders.pkl", _p("fibrosis_label_encoders.pkl")
)
FIBROSIS_SCALER_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\fibrosis_standard_scaler.pkl", _p("fibrosis_standard_scaler.pkl"))

STEATOSIS_LABEL_ENCODERS_PATH = os.getenv(
    "C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\steatosis_label_encoders.pkl", _p("label_encoders(stage_steatosis).pkl")
)
STEATOSIS_SCALER_PATH = os.getenv("C:\\Users\\dell\\Downloads\\Masald\\MASLD AI Detection Website\\Backend\\app\\models\\standard_scaler(stage_steatosis).pkl", _p("standard_scaler(stage_steatosis).pkl"))


IMG_HEIGHT = 224
IMG_WIDTH = 224
MASK_THRESHOLD = 0.5 

GRADES_CLASS_NAMES = ["Grade_1", "Grade_2", "Grade_3"]

FIBROSIS_IMAGE_CLASS_NAMES = [
    "F0-F1 (No/minimal fibrosis)",
    "F2-F3 (Significant fibrosis)",
    "F3-F4 (Advanced fibrosis)",
]

STEATOSIS_IMAGE_CLASS_NAMES = [
    "S0 (Normal)",
    "S1 (Mild)",
    "S2 (Moderate)",
    "S3 (Severe)",
]


TABULAR_FEATURE_ORDER = [
    "Age",
    "DM",
    "HTN",
    "BMI",
    "waist circumference (cm)",
    "triglceride",
    "HDL",
    "cholesterol",
    "INR",
    "FBG",
    "HBAIC",
    "ALT",
    "AST",
    "Albumin",
    "Bilubin",
    "weight in KG",
    "height in cm ",
    "Height *2",
    "Height in meter ",
    "MASLD",
]

TABULAR_NUMERIC_COLS = [
    "Age",
    "BMI",
    "waist circumference (cm)",
    "triglceride",
    "HDL",
    "cholesterol",
    "INR",
    "FBG",
    "HBAIC",
    "ALT",
    "AST",
    "Albumin",
    "weight in KG",
    "height in cm ",
    "Height *2",
    "Height in meter ",
]

TABULAR_CATEGORICAL_COLS = ["DM", "HTN", "Bilubin", "MASLD"]

FIBROSIS_TARGET_COL = "stage_fibrosis"
STEATOSIS_TARGET_COL = "stage_steatosis"
