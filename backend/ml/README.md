# ml/ — Maskininlärningsmodeller

## Katalogstruktur

```
ml/
├── models/
│   ├── genre_classifier.pkl   ← tränad RF-modell (skapas av train_genre_model.py)
│   └── model_info.json        ← metadata, accuracy, feature importance
└── train_genre_model.py       ← träningsskript
```

## Träna modellen

### Steg 1 — Installera beroenden

```bash
pip install scikit-learn
# eller uppdatera requirements.txt och kör:
pip install -r backend/requirements.txt
```

### Steg 2A — Med GTZAN (rekommenderat för produktion)

GTZAN Dataset: 1000 låtar, 10 genrer, 30 sek/klipp, ~1.2 GB

**Via Kaggle:**
```bash
pip install kaggle
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d data/
python ml/train_genre_model.py --data-dir data/genres_original
```

**Direkt från marsyas.info:**
```bash
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xf genres.tar.gz -C data/
python ml/train_genre_model.py --data-dir data/genres
```

### Steg 2B — Syntetisk bootstrap (för CI/CD och snabbtestning)

```bash
python ml/train_genre_model.py --synthetic
# Eller med fler samples för bättre modell:
python ml/train_genre_model.py --synthetic --n-synthetic 500
```

> ⚠️ Syntetisk data ger ~85% CV accuracy men sämre generalisering på riktiga låtar.
> Använd GTZAN för produktionsbruk.

## Typiska resultat

| Datakälla        | CV Accuracy | Notering                     |
|------------------|-------------|------------------------------|
| GTZAN (100 låtar/genre) | ~75-82% | Bäst för produktion         |
| Syntetisk 200/genre     | ~85%    | Verifierar pipeline, ej prod |

## Feature-vektor (48 dimensioner)

| Index | Feature                |
|-------|------------------------|
| 0-12  | MFCC mean (13 koeff.)  |
| 13-25 | MFCC std               |
| 26-37 | Chroma mean (12 noter) |
| 38    | BPM                    |
| 39    | Spectral centroid mean |
| 40    | Spectral rolloff mean  |
| 41    | Spectral bandwidth     |
| 42    | Spectral contrast      |
| 43    | Zero crossing rate     |
| 44    | Energy (RMS)           |
| 45    | Danceability           |
| 46    | Beat strength          |
| 47    | Rhythmic complexity    |

## Genrer

GTZAN-genrerna mappas till plattformens genrenamn:

| GTZAN      | Platform     |
|------------|--------------|
| blues      | blues        |
| classical  | classical    |
| country    | country      |
| disco      | electronic   |
| hiphop     | hip-hop      |
| jazz       | jazz         |
| metal      | rock         |
| pop        | pop          |
| reggae     | reggae       |
| rock       | rock         |

## Uppdatera modellen

```bash
# Träna om och deploy:
python ml/train_genre_model.py --data-dir data/genres_original
# Modellen laddas automatiskt av GenreClassifier nästa gång servern startar.
# Ingen kodändring krävs.
```
