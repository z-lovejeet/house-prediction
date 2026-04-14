import os

CORE_DIR = os.path.dirname(__file__)
APP_DIR = os.path.dirname(CORE_DIR)
BACKEND_DIR = os.path.dirname(APP_DIR)

PROCESSED_DIR = os.path.join(BACKEND_DIR, "ml_pipeline", "outputs", "processed")
OPT_DIR = os.path.join(BACKEND_DIR, "ml_pipeline", "outputs", "optimization")
