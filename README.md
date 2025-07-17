# FineTune

Ein Framework zum Finetuning von Safeguard-Modellen für die CLI-Assistenzsystem-Applikation **terminAl**. Dieses Projekt erstellt ein spezialisiertes Encoder-Modell zur Risikobewertung von Linux/Bash-Befehlen und SQL-Abfragen.

## Projektübersicht

FineTune implementiert eine vollständige Pipeline zur Erstellung eines Safeguard-Modells, das Terminal-Befehle in Risikokategorien (`low`, `medium`, `high`) klassifiziert. Das trainierte Modell wird in der terminAl-Applikation verwendet, um Benutzer vor potenziell gefährlichen Befehlen zu warnen.

> ⚠️ **WARNUNG**: Dieses Modell wurde für einen Proof of Concept (PoC) trainiert und sollte keinesfalls in einer produktiven Umgebung eingesetzt werden. Für den Produktiveinsatz sind eine umfassende Evaluierung, weitere Optimierung und ausführliche Tests erforderlich.

## Projektstruktur

```
FineTune/
├── data_augmentation.ipynb         # Jupyter Notebook für synthetische Datengenerierung
├── data_augmentation.py           # Python-Script für synthetische Datengenerierung
├── finetune_encoder.ipynb         # Jupyter Notebook für Modell-Finetuning
├── finetune_encoder.py            # Python-Script für Modell-Finetuning
├── load_and_clean_data.ipynb      # Jupyter Notebook für Datensammlung und -bereinigung
├── load_and_clean_data.py         # Python-Script für Datensammlung und -bereinigung
├── LICENSE                        # Lizenzinformationen
├── README.md                      # Diese Datei
└── requirements.txt               # Python-Abhängigkeiten
```

## Installation

1. **Repository klonen:**
```bash
git clone <repository-url>
cd FineTune
```

2. **Virtuelle Umgebung erstellen:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

3. **Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

4. **Umgebungsvariablen einrichten:**
```bash
# .env Datei erstellen
echo "HF_TOKEN=<huggingface_token>" > .env
```

## Verwendung

Das Projekt folgt einer dreistufigen Pipeline:

### 1. Datensammlung und -bereinigung

Sammelt und bereinigt Daten von verschiedenen Hugging Face Datasets für Linux/Bash-Befehle und SQL-Abfragen.

```bash
# Als Python-Script ausführen
python load_and_clean_data.py

# Oder als Jupyter Notebook
jupyter notebook load_and_clean_data.ipynb
```

**Verwendete Datasets:**
- [`AnishJoshi/nl2bash-custom`](https://huggingface.co/datasets/AnishJoshi/nl2bash-custom) (Linux/Bash-Befehle, Lizenz: Nicht spezifiziert)
- [`aelhalili/bash-commands-dataset`](https://huggingface.co/datasets/aelhalili/bash-commands-dataset) (Linux/Bash-Befehle, Lizenz: MIT)
- [Kushagragoyal060705/Complex-Linux-commands-from-natual-language](https://www.kaggle.com/datasets/kushagragoyal060705/complex-linux-commands-from-natual-language) (Linux/Bash-Befehle, Lizenz: Apache 2.0)
- [`omeryentur/text-to-postgresql`](https://huggingface.co/datasets/omeryentur/text-to-postgresql) (SQL-Abfragen, Lizenz: Nicht spezifiziert)
- [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Synthetische SQL-Daten, Lizenz: Apache 2.0)

### 2. Data Augmentation (Synthetische Datengenerierung)

Generiert synthetische Trainingsdaten mit Risikobewertungen unter Verwendung von Ollama/Llama.

```bash
# Als Python-Script ausführen
python data_augmentation.py

# Oder als Jupyter Notebook
jupyter notebook data_augmentation.ipynb
```

**Voraussetzungen:**
- Ollama muss installiert und laufend sein
- Ein kompatibles Llama-Modell muss verfügbar sein

### 3. Modell-Finetuning

Führt das Finetuning eines DeBERTa v3 base Modells für die Risikokategorisierung durch.

```bash
# Als Python-Script ausführen
python finetune_encoder.py

# Oder als Jupyter Notebook
jupyter notebook finetune_encoder.ipynb
```

## Modelldetails

- **Basis-Modell:** [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base)
- **Aufgabe:** Sequenzklassifikation (3 Klassen)
- **Klassen:** 
  - `0: low` - Geringe Risikobefehle (nur Lesezugriff)
  - `1: medium` - Mittlere Risikobefehle (reversible Änderungen)
  - `2: high` - Hohe Risikobefehle (irreversible/systemkritische Operationen)

## Konfiguration

Die Trainingskonfiguration kann in `finetune_encoder.py` angepasst werden:

```python
TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    max_steps=1000,
    learning_rate=2e-4,
    # ... weitere Parameter
)
```

## Ausgabe

Das trainierte Modell wird automatisch zu Hugging Face Hub hochgeladen:
- **Modell**: [`terminAl-thesis-2025/deberta-v3-base-terminAl-guard`](https://huggingface.co/terminAl-thesis-2025/deberta-v3-base-terminAl-guard)
- **Tokenizer**: [`terminAl-thesis-2025/deberta-v3-base-terminAl-guard`](https://huggingface.co/terminAl-thesis-2025/deberta-v3-base-terminAl-guard)


## Anforderungen

- Python 3.8+
- CUDA-kompatible GPU (empfohlen für Finetuning)
- Ollama (für Data Augmentation)
- Hugging Face Account und Token
- Mindestens 8GB VRAM für Finetuning

## Lizenz

Siehe [LICENSE](LICENSE) Datei für Details.

## Dependencies: 
Dieses Projekt nutzt verschiedene Python-Pakete (siehe `requirements.txt`), die unter ihren jeweiligen Lizenzen stehen. Bei der Nutzung der Dependencies sind deren Lizenzbedingungen zu beachten.

### Verwendete Basis-Modelle

**DeBERTa v3 Base:**
Das Finetuning basiert auf [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base), welches unter der MIT-Lizenz von Microsoft veröffentlicht wurde.

### Verwendete LLM-Modelle
Dieses Projekt wurde mit Meta Llama-Modellen (Llama 3.1 und Llama 3.2) entwickelt und getestet. Bei der Verwendung von Llama-Modellen gelten die entsprechenden Lizenzbedingungen von Meta:

- **Llama 3.1:** Unterliegt der Llama 3.1 Community License von Meta
- **Llama 3.2:** Unterliegt der Llama 3.2 Community License von Meta

Für kommerzielle Nutzung oder bei mehr als 700 Millionen monatlich aktiven Nutzenden sind separate Lizenzvereinbarungen mit Meta erforderlich. Weitere Informationen und die vollständigen Lizenzbedingungen finden sich auf der Meta Llama Website.

### Dataset-Lizenzen

Dieses Projekt verwendet Daten aus verschiedenen Quellen mit unterschiedlichen Lizenzen:

- **MIT License**: `aelhalili/bash-commands-dataset`
- **Apache 2.0**: ` Kushagragoyal060705/Complex-Linux-commands-from-natual-language`, `gretelai/synthetic_text_to_sql`
- **Nicht spezifiziert**: `AnishJoshi/nl2bash-custom`, `omeryentur/text-to-postgresql`

**Hinweis**: Bitte überprüfe die Lizenzkompatibilität vor der kommerziellen Nutzung. Für Datasets ohne spezifizierte Lizenz, bitte an die jeweiligen Autoren wenden.