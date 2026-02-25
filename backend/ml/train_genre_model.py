"""
Tränar en Random Forest-modell för genreklassificering.

Datakällor (i prioritetsordning):
  1. GTZAN dataset — 1000 låtar, 10 genrer, 30 sek per klipp
     Ladda ner: http://marsyas.info/downloads/datasets.html
     Eller via: pip install kaggle && kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification

  2. FMA (Free Music Archive) — om GTZAN ej tillgängligt
     https://github.com/mdeff/fma

  3. Syntetisk bootstrap — för CI/CD och snabbtestning
     Genererar fake feature-vektorer med rimliga distributions per genre.
     Ge INTE produktionskvalitet men verifierar att pipeline fungerar.

Användning:
  # Med GTZAN-data (rekommenderat):
  python train_genre_model.py --data-dir /path/to/gtzan/genres_original

  # Syntetisk bootstrap (ingen data krävs):
  python train_genre_model.py --synthetic

  # Med FMA small:
  python train_genre_model.py --data-dir /path/to/fma_small --format fma

Output:
  ml/models/genre_classifier.pkl  — tränad RF-modell
  ml/models/model_info.json       — metadata, accuracy, confusion matrix
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Konstanter — måste matcha features.py exakt ────────────────────────────
FEATURE_DIM = 48   # 13 MFCC mean + 13 MFCC std + 12 chroma mean + 10 spektrala/rytmiska

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# Mapping GTZAN -> våra genrenamn (i klassificeraren)
GENRE_MAP = {
    "blues":     "blues",
    "classical": "classical",
    "country":   "country",
    "disco":     "electronic",
    "hiphop":    "hip-hop",
    "jazz":      "jazz",
    "metal":     "rock",
    "pop":       "pop",
    "reggae":    "reggae",
    "rock":      "rock",
}

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "genre_classifier.pkl"
INFO_PATH  = MODEL_DIR / "model_info.json"


# ── Feature-extraktion från GTZAN-filer ────────────────────────────────────

def extract_features_from_file(filepath: str) -> Optional[np.ndarray]:
    """
    Extraherar feature-vektor från en ljudfil.
    Återanvänder features.py-logiken för konsistens med produktionen.
    """
    # Lägg till backend/ i path så vi hittar app.core.features
    backend_dir = Path(__file__).parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        from app.core.features import extract
        with open(filepath, "rb") as fh:
            raw = fh.read()
        features = extract(raw, Path(filepath).name)
        fv = features.get("feature_vector")
        if fv and len(fv) == FEATURE_DIM:
            return np.array(fv, dtype=np.float32)
        else:
            print(f"  VARNING Fel dimension: {len(fv) if fv else 'None'} (förväntat {FEATURE_DIM})")
            return None
    except Exception as e:
        print(f"  FEL {Path(filepath).name}: {e}")
        return None


def load_gtzan(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Laddar GTZAN-dataset från katalogstruktur:
      data_dir/
        blues/
          blues.00000.wav
          ...
        classical/
          ...
    """
    data_path = Path(data_dir)
    X, y = [], []
    total, failed = 0, 0

    for genre_dir in sorted(data_path.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre_name = genre_dir.name.lower()
        if genre_name not in GTZAN_GENRES:
            print(f"  Skippar okänd katalog: {genre_name}")
            continue

        mapped_genre = GENRE_MAP.get(genre_name, genre_name)
        files = list(genre_dir.glob("*.wav")) + list(genre_dir.glob("*.mp3"))
        print(f"\n{genre_name} -> {mapped_genre} ({len(files)} filer)")

        for fpath in files:
            total += 1
            fv = extract_features_from_file(str(fpath))
            if fv is not None:
                X.append(fv)
                y.append(mapped_genre)
                if len(X) % 20 == 0:
                    print(f"  {len(X)} klara...", end="\r")
            else:
                failed += 1

    print(f"\nOK Laddade {len(X)}/{total} filer ({failed} misslyckades)")
    return np.array(X, dtype=np.float32), np.array(y)


# ── Syntetisk bootstrap ────────────────────────────────────────────────────

# Typiska feature-profiler per genre (approximativa medelvärden och std)
# Ordning: 13 MFCC mean, 13 MFCC std, 12 chroma mean, bpm, spec_centroid,
#          spec_rolloff, spec_bandwidth, spec_contrast, zcr, energy,
#          danceability, beat_strength, rhythmic_complexity
GENRE_PROFILES = {
    "electronic": dict(
        bpm_mean=128, bpm_std=10,
        energy_mean=0.08, energy_std=0.02,
        zcr_mean=0.04, zcr_std=0.01,
        spec_centroid=4000, spec_centroid_std=500,
        danceability=0.80, dance_std=0.10,
    ),
    "rock": dict(
        bpm_mean=130, bpm_std=20,
        energy_mean=0.07, energy_std=0.02,
        zcr_mean=0.08, zcr_std=0.02,
        spec_centroid=3500, spec_centroid_std=600,
        danceability=0.55, dance_std=0.15,
    ),
    "pop": dict(
        bpm_mean=115, bpm_std=15,
        energy_mean=0.05, energy_std=0.01,
        zcr_mean=0.05, zcr_std=0.01,
        spec_centroid=3000, spec_centroid_std=400,
        danceability=0.70, dance_std=0.12,
    ),
    "hip-hop": dict(
        bpm_mean=90, bpm_std=10,
        energy_mean=0.04, energy_std=0.01,
        zcr_mean=0.03, zcr_std=0.01,
        spec_centroid=2000, spec_centroid_std=300,
        danceability=0.75, dance_std=0.12,
    ),
    "jazz": dict(
        bpm_mean=130, bpm_std=30,
        energy_mean=0.03, energy_std=0.01,
        zcr_mean=0.04, zcr_std=0.015,
        spec_centroid=2500, spec_centroid_std=500,
        danceability=0.50, dance_std=0.15,
    ),
    "classical": dict(
        bpm_mean=100, bpm_std=30,
        energy_mean=0.02, energy_std=0.01,
        zcr_mean=0.015, zcr_std=0.008,
        spec_centroid=2000, spec_centroid_std=600,
        danceability=0.30, dance_std=0.15,
    ),
    "blues": dict(
        bpm_mean=90, bpm_std=15,
        energy_mean=0.035, energy_std=0.01,
        zcr_mean=0.04, zcr_std=0.012,
        spec_centroid=2200, spec_centroid_std=400,
        danceability=0.55, dance_std=0.15,
    ),
    "country": dict(
        bpm_mean=110, bpm_std=15,
        energy_mean=0.04, energy_std=0.01,
        zcr_mean=0.045, zcr_std=0.012,
        spec_centroid=2800, spec_centroid_std=400,
        danceability=0.60, dance_std=0.12,
    ),
    "reggae": dict(
        bpm_mean=80, bpm_std=10,
        energy_mean=0.035, energy_std=0.01,
        zcr_mean=0.03, zcr_std=0.01,
        spec_centroid=1800, spec_centroid_std=300,
        danceability=0.72, dance_std=0.10,
    ),
}

RNG = np.random.default_rng(42)


def _synthetic_vector(p: dict) -> np.ndarray:
    """Genererar en feature-vektor baserat på en genres profil."""
    bpm          = RNG.normal(p["bpm_mean"], p["bpm_std"])
    energy       = max(0.001, RNG.normal(p["energy_mean"], p["energy_std"]))
    zcr          = max(0.001, RNG.normal(p["zcr_mean"], p["zcr_std"]))
    spec_c       = max(500,   RNG.normal(p["spec_centroid"], p["spec_centroid_std"]))
    danceability = float(np.clip(RNG.normal(p["danceability"], p["dance_std"]), 0, 1))

    # 13 MFCC mean — korrelerade med spectral shape
    mfcc_base   = np.array([-100, 50, 30, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1], dtype=float)
    mfcc_scale  = spec_c / 2000
    mfcc_mean   = mfcc_base * mfcc_scale + RNG.normal(0, 5, 13)

    # 13 MFCC std — korrelerade med dynamik
    mfcc_std    = np.abs(mfcc_mean * 0.3) + RNG.normal(0, 2, 13)

    # 12 chroma mean — jämn fördelning med liten variation
    chroma_mean = RNG.dirichlet(np.ones(12) * 2)

    # Spektrala features
    spec_rolloff   = spec_c * 2.5 + RNG.normal(0, 200)
    spec_bandwidth = spec_c * 0.5 + RNG.normal(0, 100)
    spec_contrast  = 20 + RNG.normal(0, 5)
    beat_strength  = float(np.clip(danceability * 0.8 + RNG.normal(0, 0.1), 0, 1))
    rhythmic_comp  = float(np.clip(0.5 + RNG.normal(0, 0.15), 0, 1))

    vec = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean,
        [bpm, spec_c, spec_rolloff, spec_bandwidth, spec_contrast,
         zcr, energy, danceability, beat_strength, rhythmic_comp]
    ])
    assert len(vec) == FEATURE_DIM, f"Fel dimension: {len(vec)}"
    return vec.astype(np.float32)


def generate_synthetic_data(n_per_genre: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Genererar syntetisk träningsdata för bootstrap."""
    X, y = [], []
    for genre, profile in GENRE_PROFILES.items():
        for _ in range(n_per_genre):
            X.append(_synthetic_vector(profile))
            y.append(genre)
    print(f"OK Genererade {len(X)} syntetiska samples ({len(GENRE_PROFILES)} genrer, {n_per_genre}/genre)")
    return np.array(X, dtype=np.float32), np.array(y)


# ── Träning ────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, synthetic: bool = False) -> dict:
    """Tränar Random Forest, returnerar modell + metrics."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n{'='*50}")
    print(f"Tränar modell")
    print(f"  Samples:    {len(X)}")
    print(f"  Dimensioner: {X.shape[1]}")
    print(f"  Genrer:     {sorted(set(y))}")
    print(f"  Syntetisk:  {synthetic}")
    print(f"{'='*50}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: StandardScaler + RandomForest
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",   # hantera obalanserade klasser
            random_state=42,
            n_jobs=-1,
        ))
    ])

    print("Tränar Random Forest (300 träd)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"OK Färdig på {elapsed:.1f}s")

    # Utvärdering
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    print("Kör 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    rf = model.named_steps["clf"]
    feature_names = (
        [f"mfcc_mean_{i}" for i in range(13)] +
        [f"mfcc_std_{i}"  for i in range(13)] +
        [f"chroma_{i}"    for i in range(12)] +
        ["bpm", "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
         "spectral_contrast", "zcr", "energy", "danceability", "beat_strength", "rhythmic_complexity"]
    )
    importances = rf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 viktigaste features:")
    for name, imp in top_features:
        bar = "█" * int(imp * 100)
        print(f"  {name:<30} {imp:.4f} {bar}")

    return {
        "model": model,
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "genres": sorted(set(y)),
        "feature_dim": int(X.shape[1]),
        "n_samples": int(len(X)),
        "synthetic": synthetic,
        "top_features": [(n, float(i)) for n, i in top_features],
    }


def save_model(result: dict):
    """Sparar pkl och metadata JSON."""
    import pickle
    from datetime import datetime, timezone

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result["model"], f, protocol=5)
    print(f"\nOK Modell sparad: {MODEL_PATH}")

    info = {
        "model_version": "rf-v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": result["accuracy"],
        "cv_accuracy_mean": result["cv_mean"],
        "cv_accuracy_std": result["cv_std"],
        "genres": result["genres"],
        "feature_dim": result["feature_dim"],
        "n_samples": result["n_samples"],
        "synthetic_data": result["synthetic"],
        "top_features": result["top_features"],
        "note": (
            "Tränad på syntetisk data — ersätt med GTZAN för produktionskvalitet."
            if result["synthetic"]
            else "Tränad på GTZAN-data."
        ),
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"OK Metadata sparad: {INFO_PATH}")

    print(f"\n{'='*50}")
    print(f"SAMMANFATTNING")
    print(f"  Test accuracy:    {result['accuracy']:.1%}")
    print(f"  CV accuracy:      {result['cv_mean']:.1%} ± {result['cv_std']:.1%}")
    print(f"  Genrer:           {', '.join(result['genres'])}")
    print(f"  Syntetisk data:   {result['synthetic']}")
    print(f"{'='*50}\n")

    if result["synthetic"]:
        print("VARNING  VARNING: Modellen är tränad på syntetisk data.")
        print("   Träffsäkerheten på riktiga låtar kan vara låg.")
        print("   Ladda ner GTZAN och kör om för produktionsbruk:\n")
        print("   kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification")
        print("   python train_genre_model.py --data-dir /path/to/genres_original\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Träna genre-klassificerare")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Sökväg till GTZAN genres_original/ eller FMA-katalog"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generera syntetisk träningsdata (ingen riktig data krävs)"
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=200,
        help="Antal syntetiska samples per genre (default: 200)"
    )
    parser.add_argument(
        "--format", choices=["gtzan", "fma"], default="gtzan",
        help="Dataformat (default: gtzan)"
    )
    args = parser.parse_args()

    # Välj datakälla
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"FEL Katalogen finns inte: {args.data_dir}")
            sys.exit(1)
        print(f"Laddar {args.format.upper()}-data från: {args.data_dir}")
        X, y = load_gtzan(args.data_dir)
        synthetic = False
    elif args.synthetic:
        print("Genererar syntetisk data...")
        X, y = generate_synthetic_data(n_per_genre=args.n_synthetic)
        synthetic = True
    else:
        # Auto: kolla om GTZAN finns på vanliga platser
        candidates = [
            "./data/genres_original",
            "./data/gtzan/genres_original",
            os.path.expanduser("~/datasets/gtzan/genres_original"),
        ]
        found = next((p for p in candidates if os.path.isdir(p)), None)
        if found:
            print(f"Hittade GTZAN-data: {found}")
            X, y = load_gtzan(found)
            synthetic = False
        else:
            print("Ingen data hittad — kör med --synthetic eller --data-dir")
            print("Startar i syntetiskt läge för demo...\n")
            X, y = generate_synthetic_data(n_per_genre=args.n_synthetic)
            synthetic = True

    if len(X) == 0:
        print("FEL Inga features extraherades. Kontrollera datakatalogen.")
        sys.exit(1)

    result = train_model(X, y, synthetic=synthetic)
    save_model(result)


if __name__ == "__main__":
    main()
