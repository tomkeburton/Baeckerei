# Bäckerei Umsatzprognose mit Temporal Fusion Transformer (TFT)

Zeitreihen-Forecasting für Bäckerei-Umsatzdaten mit Temporal Fusion Transformer, orchestriert über Cadence Workflow.

## 📊 Projekt-Übersicht

Dieses Projekt prognostiziert Umsätze für verschiedene Warengruppen einer Bäckerei unter Berücksichtigung von:
- Saisonalität und Trends
- Feiertagen (deutsche Bundesfeiertage)
- Schulferien (Schleswig-Holstein)
- Lokale Events (Kieler Woche)
- Wetterdaten (Temperatur, Niederschlag, Sonnenscheindauer)

**Datenzeitraum:**
- Training: 2013-2017 (5 Jahre)
- Test: 2018 (1 Jahr)
- Warengruppen: 1 (Kuchen) und 2

## 🗂️ Projektstruktur

```
Baeckerei/
├── data/
│   ├── raw/                           # Original-Daten
│   │   └── Umsatz_Baeckerei.csv       # Vollständiger Datensatz
│   ├── processed/                     # Gefilterte Daten
│   │   ├── train_data.csv             # 2013-2017, Warengruppe 1+2
│   │   └── test_data.csv              # 2018, Warengruppe 1+2
│   └── external_factors/              # Externe Einflussfaktoren
│       ├── german_holidays.csv        # Deutsche Feiertage
│       ├── schleswig_holstein_vacations.csv  # SH Schulferien
│       ├── kieler_woche.csv           # Kieler Woche Events
│       ├── weather_data/              # Wetterdaten
│       │   ├── Temp.csv
│       │   ├── Niederschlag.csv
│       │   └── Sonnenscheindauer.csv
│       ├── ferien_sh_original.csv     # Original Feriendaten
│       └── kieler_woche_original.csv  # Original Event-Daten
├── scripts/
│   ├── cadence_tft_workflow.py        # Haupt-Workflow für Cadence
│   └── data_analysis.py               # Datenanalyse-Skript (geplant)
├── notebooks/                         # Jupyter Notebooks
├── results/
│   ├── figures/                       # Visualisierungen
│   └── reports/                       # Analyse-Reports
├── models/                            # Trainierte Modelle
├── requirements.txt                   # Python Dependencies
└── README.md                          # Diese Datei
```

## 🚀 Setup

### 1. Repository klonen
```bash
git clone <repository-url>
cd Baeckerei
```

### 2. Virtual Environment erstellen
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# oder
.venv\Scripts\activate     # Windows
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Cadence Server starten
```bash
# Cadence muss auf localhost:7233 laufen
# Siehe: https://cadenceworkflow.io/docs/get-started/
```

## 📈 Verwendung

### Datenanalyse durchführen
```bash
python scripts/data_analysis.py
```

### TFT-Workflow ausführen
```bash
python scripts/cadence_tft_workflow.py
```

## 📊 Datensatz-Details

**Bekannte Besonderheiten:**
- ✅ Keine fehlenden Werte in Umsatzdaten
- ⚠️ Fehlende Tage (Feiertage/Geschlossen) müssen behandelt werden
- 📈 Kuchen (Warengruppe 1): Extreme Min-Max-Spreizung
- 🎄 Massiver Jahresend-Spike bei Kuchen (Q4)
- ⚠️ Mögliche Artefakte in Mai 2014

## 🔧 Technologie-Stack

- **Forecasting Model:** Temporal Fusion Transformer (TFT)
- **Deep Learning:** PyTorch, PyTorch Lightning, PyTorch Forecasting
- **Workflow Orchestration:** Cadence (Temporalio)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Analysis:** SciPy, Statsmodels, Scikit-learn

## 👥 Zusammenarbeit

Dieses Projekt ist für Teamarbeit konzipiert. Workflow:

1. Branch für neue Features erstellen
2. Changes committen und pushen
3. Pull Request erstellen
4. Code Review durch Team-Mitglieder
5. Merge in main branch

## 📝 Nächste Schritte

- [ ] Detaillierte Datenanalyse durchführen
- [ ] Feature Engineering (Feiertags-Flags, Ferien-Flags)
- [ ] Fehlende Tage behandeln
- [ ] Outlier-Analyse und -Behandlung
- [ ] TFT-Modell optimieren
- [ ] Hyperparameter-Tuning
- [ ] Ergebnisse dokumentieren

## 📄 Lizenz

Uni-Projekt - Alle Rechte vorbehalten
