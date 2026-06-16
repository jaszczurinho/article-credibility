# Wiarygodność Artykułów Prasowych

Projekt analizuje wiarygodność artykułów prasowych przy użyciu technik uczenia maszynowego. Na podstawie cech tekstowych, lingwistycznych i emocjonalnych artykułów budowane są modele klasyfikacyjne pozwalające ocenić, czy dany artykuł pochodzi z wiarygodnego źródła.

## Struktura projektu

```
article-credibility/
├── data/               # Surowe i przetworzone dane
├── input/              # Pliki wejściowe do predykcji (CSV)
├── models/             # Zapisane wytrenowane modele (.joblib)
├── notebooks/          # Notebooki Jupyter
│   ├── data_overview.ipynb             # Przegląd i eksploracja danych
│   ├── features_EDA.ipynb              # Analiza eksploracyjna cech (EDA)
│   ├── modelling.ipynb                 # Budowanie i ocena modeli klasyfikacyjnych
│   └── modelling_embeddings.ipynb      # Modelowanie z wykorzystaniem embeddingów
└── src/
    ├── data/
    │   ├── articles_scraper.py         # Pobieranie artykułów przez NewsAPI
    │   ├── create_features.py          # Tworzenie cech lingwistycznych (spaCy + NER + POS)
    │   └── emotion_extraction.py       # Ekstrakcja cech emocjonalnych (DistilRoBERTa)
    └── prediction/
        └── prediction_pipeline.py      # Potok predykcji dla nowych artykułów
```

## Opis skryptów (`src/`)

### `src/data/articles_scraper.py`
Pobiera artykuły z NewsAPI dla zdefiniowanych zapytań tematycznych (np. klimat, AI, polityka). Artykuły są pobierane ze z góry określonych domen – osobno dla źródeł **rzetelnych** (np. Reuters, Guardian, NPR) i **nierzetelnych** (np. Breitbart, ZeroHedge). Wynikowe dane zapisywane są do plików CSV w `data/raw/`.

Wymagana zmienna środowiskowa: `NEWSAPI_KEY`.

### `src/data/emotion_extraction.py`
Przy użyciu modelu `j-hartmann/emotion-english-distilroberta-base` (Hugging Face) oblicza cechy emocjonalne artykułu: prawdopodobieństwa 7 emocji (gniew, wstręt, strach, radość, neutralność, smutek, zaskoczenie) dla pierwszego fragmentu tekstu oraz entropię emocjonalną całego artykułu. Długie teksty są automatycznie dzielone na fragmenty mieszczące się w oknie kontekstowym modelu.

### `src/data/create_features.py`
Tworzy cechy lingwistyczne z użyciem modelu `en_core_web_sm` biblioteki spaCy:
- **NER** – zagęszczenie encji named entity: osoby, organizacje, miejsca, liczby
- **POS** – zagęszczenie zaimków (1., 2., 3. osoba), przymiotników, spójników
- **Cudzysłowy** – zagęszczenie znaków cytowania w tekście

Dane wejściowe: `data/processed/final_dataset.csv`. Wynik: `data/processed/final_dataset_with_features.csv`.

### `src/prediction/prediction_pipeline.py`
Gotowy potok predykcji dla nowych artykułów – szczegóły poniżej w sekcji [Predykcja](#predykcja).

---

## Instalacja

Projekt używa [`uv`](https://github.com/astral-sh/uv) do zarządzania środowiskiem i zależnościami.

### 1. Instalacja `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Klonowanie repozytorium i instalacja zależności

```bash
git clone https://github.com/jaszczurinho/article-credibility.git
cd article-credibility

# Tworzy venv i instaluje wszystkie zależności z pyproject.toml
uv sync
```

### 3. Aktywacja środowiska wirtualnego

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

Po aktywacji prefiks `(.venv)` powinien pojawić się w terminalu.

### 4. Zmienne środowiskowe

Utwórz plik `.env` w głównym katalogu projektu (wymagane tylko do pobierania danych):

```env
NEWSAPI_KEY=twój_klucz_newsapi
```

---

## Uruchamianie notebooków

```bash
# Upewnij się, że środowisko jest aktywne, następnie uruchom Jupyter
jupyter notebook
# lub
jupyter lab
```

Notebooki należy uruchamiać w kolejności:

1. `data_overview.ipynb` – zapoznanie się z danymi
2. `features_EDA.ipynb` – analiza i inżynieria cech
3. `modelling.ipynb` – trenowanie i ewaluacja modeli
4. `modelling_embeddings.ipynb` – eksperymenty z embeddingami

---

## Predykcja

Skrypt `src/prediction/prediction_pipeline.py` pozwala ocenić wiarygodność nowych artykułów przy użyciu wytrenowanego modelu.

### Przygotowanie danych wejściowych

Utwórz plik CSV z kolumną `text`, gdzie każdy wiersz to treść jednego artykułu:

```csv
text
"Treść pierwszego artykułu..."
"Treść drugiego artykułu..."
```

Umieść plik w katalogu `input/`.

### Uruchomienie predykcji

```bash
# Automatycznie wykryje plik CSV z katalogu input/
python src/prediction/prediction_pipeline.py

# Lub wskaż plik ręcznie
python src/prediction/prediction_pipeline.py --file input/moje_artykuly.csv

# Wybór modelu (domyślnie: stacking)
python src/prediction/prediction_pipeline.py --model stacking
```

### Wynik

Plik `output/predictions.csv` z kolumnami:

| Kolumna      | Opis                                              |
|--------------|---------------------------------------------------|
| `text`       | Oryginalny tekst artykułu                         |
| `prediction` | Przewidiana etykieta: `0` = rzetelny, `1` = nierzetelny |
| `confidence` | Pewność modelu (0.0–1.0)                          |

---

## Technologie

| Biblioteka | Zastosowanie |
|---|---|
| scikit-learn | Modele klasyfikacyjne, preprocessing |
| XGBoost | Gradient boosting |
| Optuna | Optymalizacja hiperparametrów |
| pandas | Przetwarzanie danych |
| spaCy (`en_core_web_sm`) | Analiza lingwistyczna (NER, POS) |
| Transformers (DistilRoBERTa) | Ekstrakcja cech emocjonalnych |
| newspaper3k | Pobieranie i parsowanie artykułów |
| NewsAPI | Źródło danych do trenowania |
| matplotlib / seaborn | Wizualizacje |
