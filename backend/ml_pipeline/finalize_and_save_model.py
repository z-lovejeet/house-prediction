"""
Importable alias for 07_finalize_and_save_model.py
====================================================
Python module names cannot start with digits.  This shim re-exports
the public API from the Phase 7 script so that other packages
(e.g. backend.main) can do:

    from backend.ml_pipeline.finalize_and_save_model import predict_price
"""

import importlib
import os
import sys

# Ensure the ml_pipeline directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

_mod = importlib.import_module("07_finalize_and_save_model")

# Re-export the public API
predict_price = _mod.predict_price
run_finalize  = _mod.run_finalize
