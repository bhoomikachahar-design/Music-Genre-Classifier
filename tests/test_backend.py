"""
Unit and Integration Tests for Music Genre Classifier Backend
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import (
    health_check,
    list_samples,
    predict_sample,
    get_models_info,
    get_pipeline_info,
    get_history
)

def run_tests():
    print("================ BACKEND TESTS ================")
    
    # 1. Health check
    h = health_check()
    print("[TEST 1] Health Check:", h["status"])
    assert h["status"] == "healthy"
    assert len(h["models_loaded"]) >= 2
    assert len(h["classes"]) == 10

    # 2. List samples
    samples = list_samples()
    print(f"[TEST 2] Samples count: {len(samples)}")
    assert len(samples) >= 5

    # 3. Predict Classical Sample (Ensemble)
    res_c = predict_sample("classical", model="ensemble")
    print(f"[TEST 3] Classical Prediction: {res_c['prediction']['predicted_genre']} ({res_c['prediction']['confidence_percentage']}%)")
    assert res_c["status"] == "success"
    assert res_c["prediction"]["predicted_genre"] == "classical"
    assert len(res_c["visualizations"]["waveform"]) > 50
    assert res_c["visualizations"]["mfcc_heatmap"]["rows"] == 20

    # 4. Predict Rock Sample (Random Forest)
    res_r = predict_sample("rock", model="random_forest")
    print(f"[TEST 4] Rock Prediction (RF): {res_r['prediction']['predicted_genre']} ({res_r['prediction']['confidence_percentage']}%)")
    assert res_r["status"] == "success"
    assert res_r["prediction"]["predicted_genre"] == "rock"

    # 5. Predict Jazz Sample (SVM)
    res_j = predict_sample("jazz", model="svm")
    print(f"[TEST 5] Jazz Prediction (SVM): {res_j['prediction']['predicted_genre']} ({res_j['prediction']['confidence_percentage']}%)")
    assert res_j["status"] == "success"

    # 6. Models Info
    m = get_models_info()
    print(f"[TEST 6] Models Info: {list(m['models'].keys())}")
    for name, data in m["models"].items():
        print(f"         {name}: Accuracy={data['metrics']['accuracy']*100:.1f}%, Macro F1={data['metrics']['macro_f1']:.4f}")
    assert "random_forest" in m["models"]
    assert "svm" in m["models"]
    assert "ensemble" in m["models"]

    # 7. Pipeline Info
    p = get_pipeline_info()
    print(f"[TEST 7] Pipeline Stages: {len(p['stages'])} stages verified")
    assert len(p["stages"]) == 5

    # 8. History
    hist = get_history()
    print(f"[TEST 8] Analysis History Records: {len(hist)}")
    assert len(hist) >= 3

    print("================ ALL TESTS PASSED ================")

if __name__ == "__main__":
    run_tests()
