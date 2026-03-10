# TFT Modell für Warengruppe 2 (Brötchen) - Präsentation

**Modell**: Temporal Fusion Transformer (TFT)
**Ziel**: Umsatzprognose für 7 Tage in die Zukunft
**Training**: GPU in Irland via Cadence (Run 51303 - Grid Search)

---

## 1. Wie seid ihr vorgegangen?

### Was ist TFT?
**Temporal Fusion Transformer (TFT)** ist ein modernes neuronales Netzwerk speziell für Zeitreihen-Vorhersagen. Es kombiniert drei Techniken:
- **LSTM**: Versteht zeitliche Abläufe und Muster über Tage/Wochen
- **Attention**: Konzentriert sich auf wichtige Zeitpunkte (z.B. "genau vor 1 Woche war Samstag")
- **Feature Selection**: Wählt automatisch die wichtigsten Einflussfaktoren aus (z.B. Wochentag wichtiger als Wetter)

### Ansatz
- Vorhersage: 7 Tage voraus, basierend auf 28 Tagen Historie
- TFT kann Unsicherheitsbereiche berechnen (Quantile), in der Evaluation nutzen wir aber nur den Median

### Schritte
1. **Daten aufteilen**: 2016 Training, 2017 Validierung, 2018 Test
2. **TimeSeriesDataSet erstellen** (Zeitfenster) - spezielle Datenstruktur aus der PyTorch Forecasting Bibliothek, die die Daten für das TFT-Modell vorbereitet
3. **Training auf GPU in Irland via Cadence**
4. **Testen** mit echten 2018-Daten

---

## 2. Daten-Vorbereitung

### Preprocessing
- **GroupNormalizer**: Brötchen und Kuchen getrennt normalisiert (verschiedene Größenordnungen)
- **time_idx**: Durchnummerierung der Tage (TFT braucht fortlaufende Zahlen)
- **Kategoriale Features**: Als Text gespeichert für Embeddings

### Feature Engineering
- **lag_1W**: Umsatz von vor 1 Woche (wichtigstes Feature)
- **Wochentag**: Mo-So als Kategorie
- **Kalender**: Silvester, Schließtage
- TFT erzeugt automatisch weitere Hilfsvariablen

---

## 3. Zusätzliche Informationen

### Verwendete Daten
- ✅ **Ferien**: Schulferien in Schleswig-Holstein
- ✅ **Kieler Woche**: Großes lokales Event
- ❌ **Wetter**: Vorbereitet, aber **nicht verwendet** (Regen, Temperatur, Sonne)
- Getestet aber schlechtere Ergebnisse

### Zwei Arten von Features
- **Bekannt** (für Zukunft planbar): Wochentag, Feiertage, Ferien, Kieler Woche
- **Unbekannt** (nur Vergangenheit): Umsatz, lag_1W

---

## 4. Wie funktioniert das Modell?

```
28 Tage Historie → [Encoder] → [Decoder] → 7 Tage Vorhersage
```

### Hauptkomponenten
1. **Variable Selection**: Wählt automatisch die wichtigsten Features aus
2. **LSTM**: Verarbeitet zeitliche Abläufe
3. **Attention** (4 Köpfe): Konzentriert sich auf wichtige Zeitpunkte (z.B. vor 1 Woche)
4. **Quantile Loss**: Kann Unsicherheitsbereiche lernen (hier aber nur Median genutzt)

### Beste Hyperparameter (Grid Search Sieger)
- `hidden_size`: 64
- `attention_heads`: 4
- `dropout`: 0.07
- `learning_rate`: 0.012
- `batch_size`: 32
- Historie: 28 Tage → Vorhersage: 7 Tage

---

## 5. Ergebnisse (Brötchen)

### Genauigkeit auf 2018 Test-Daten

| Warengruppe | MAE (EUR) | RMSE (EUR) | MAPE (%) |
|-------------|-----------|------------|----------|
| **Brötchen**| **32.56** | **44.16**  | **8.55** |
| Kuchen      | 35.83     | 47.98      | 13.11    |
| **Gesamt**  | **34.19** | **46.11**  | **10.83**|

---

## 6. Mögliche Verbesserungen

- 🔹 **Gleitende Durchschnitte**: Über 7/14 Tage
- 🔹 **Weitere Events**: Weihnachtsmarkt, etc.
- 🔹 **Weitere Daten**: Werbekampagnen, Konkurrenz, Preise



