from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = np.load(ROOT / "data" / "MAIN_4MODEL_PREDICTIONS.npz")
Y = D["Ytrue"]
expected = {
    "RandomForest": (97.31243, 242.66293),
    "RNN": (87.83278, 225.07967),
    "LSTM": (90.10046, 230.95056),
    "HINN": (84.92431, 207.21753),
}
print("Saved test-set verification (1453 samples, 30-step/5-min horizon)")
print("Model           ADE (m)    FDE (m)    status")
print("------------------------------------------------")
ok = True
for name, (ea, ef) in expected.items():
    e = np.linalg.norm(D[name] - Y, axis=2)
    ade = float(e.mean())
    fde = float(e[:, -1].mean())
    passed = abs(ade-ea) < 1e-3 and abs(fde-ef) < 1e-3
    ok &= passed
    print(f"{name:14s} {ade:8.2f}   {fde:8.2f}    {'PASS' if passed else 'FAIL'}")
print("------------------------------------------------")
print("OVERALL:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
