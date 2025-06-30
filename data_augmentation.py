#%% md
# # _____________________
# #          Achtung bei "Run All"
# # _____________________
#%% md
# # Erstellen des Batch-Sets
#%%
# Für diverse Datenoperationen
import pandas as pd
from accelerate.commands.config.update import description

# Zum Laden von Datasets von Huggingface
from datasets import load_dataset, Dataset as DS

import requests as r

import json

from tqdm import tqdm

#%% md
# Laden des Datasets
#%%
dataset = load_dataset("terminAl-thesis-2025/combined_dataset")
dataset_df = pd.DataFrame(dataset["train"])

dataset_df
#%% md
# Hinzufügen einer eindeutigen ID (sequenzielle ID)
#%%
dataset_df["id"] = (dataset_df.index +1).astype(str)
#%%
dataset_df.head(2)
#%% md
# Definition des Systemprompts
#%%
base_prompt = '''Du bist ein spezialisierter Assistent, der natürlichsprachliche Anfragen zu CLI-Befehlen (Command Line Interface) oder SQL-Abfragen in ein strukturiertes JSON-Format übersetzt. Deine Antwort muss immer exakt im folgenden JSON-Format sein:

{{
 "command": "der vollständige und korrekte Befehl/die SQL-Abfrage",
 "tool": "bash/sql",
 "risk_level": "low/medium/high",
 "description": "Kurze Beschreibung des Befehls oder der Abfrage",
 "detailed_description": "Längere Erklärung, was der Befehl tut und wie die Parameter funktionieren",
 "potential_consequences": ["mögliche Auswirkungen oder Risiken des Befehls", "weitere Auswirkung wenn zutreffend"]
}}

Konvertiere die folgende natürlichsprachliche Anfrage in das JSON-Format. Verwende dabei ausschließlich die Informationen aus der Anfrage und dem zugehörigen Befehl:

ANFRAGE: "{0}"
BEFEHL: "{1}"

Richtlinien:
1. Bestimme den Wert für "tool" korrekt basierend auf dem Befehl: Verwende "sql" für SQL-Abfragen und "bash" für Linux/Bash-Befehle.
2. Achte unbedingt auf die korrekte Syntax des Commands. Korrigiere den Command falls nötig (z.B. fehlendes Semikolon bei SQL-Abfragen hinzufügen).
3. Gib für SQL-Abfragen "low" als risk_level an, es sei denn, es handelt sich um DELETE, DROP, ALTER oder UPDATE-Operationen.
4. Für CLI-Befehle bewerte das Risiko basierend auf der Möglichkeit von Datenverlust oder Systemänderungen.
5. Die "description" sollte prägnant sein (max. 1 Satz).
6. Die "detailed_description" sollte nicht mehr als drei Sätze umfassen.
7. Liste unter "potential_consequences" alle relevanten möglichen Auswirkungen auf, oder ein leeres Array [], wenn keine nennenswerten Risiken bestehen.
8. Antworte IMMER mit einem gültigen JSON-Objekt, ohne Einleitung oder Abschluss.
9. Verwende keine Formatierungszeichen wie "\\n", "\\t" oder andere Escape-Sequenzen im JSON.
10. Stelle sicher, dass das JSON vollständig gültig ist und direkt als Python-Dictionary geladen werden kann.
11. Halte dich strikt an doppelte Anführungszeichen für Keys und String-Werte, wie es im JSON-Standard vorgeschrieben ist.
12. Gib "description", "detailed_description" und "potential_consequences" IMMER in deutscher Sprache zurück, auch wenn die ursprüngliche Anfrage auf Englisch ist.
13. Verwende bei der Übersetzung eine klare und präzise deutsche Fachsprache.

Ergänzende Richtlinien zur Bestimmung des "risk_level":

- Verwende "low", wenn der Befehl:
  * rein lesend ist (z.B. `ls`, `cat`, `SELECT` ohne `JOIN` auf große Tabellen),
  * keine Änderungen an Daten, Systemkonfigurationen oder Dateistrukturen vornimmt,
  * typischerweise keine Auswirkungen auf andere Prozesse oder Nutzer hat.

- Verwende "medium", wenn der Befehl:
  * potenziell Änderungen an Dateien, Datenbankeinträgen oder Konfigurationen durchführt, jedoch reversibel oder mit geringer Auswirkung ist (z.B. `mv`, `cp`, `touch`, `UPDATE` mit WHERE),
  * Netzwerkverbindungen aufbaut oder Systemprozesse anstößt (z.B. `curl`, `ping`, `systemctl restart`),
  * selektive Datenbearbeitung in SQL betrifft, jedoch nicht strukturverändernd ist.

- Verwende "high", wenn der Befehl:
  * irreversible Änderungen am System oder an Daten verursacht (z.B. `rm -rf /`, `DROP TABLE`, `ALTER DATABASE`),
  * kritische Dienste beeinträchtigt oder beendet (z.B. `kill -9 1`, `shutdown`),
  * ohne weitere Rückfrage systemweite Auswirkungen haben kann,
  * Sicherheitsrisiken birgt, z.B. durch Datenfreigabe, Prozessbeeinflussung oder das Löschen von Benutzerkonten.

Beziehe immer auch Kontext und typische Folgen mit ein. Wenn Unsicherheit besteht, ist ein konservativerer (höherer) Risikowert zu bevorzugen.'''

#%% md
# Erweitere Dataframe
#%%
dataset_df["json"] = ""
#%%
dataset_df.columns
#%% md
# Erstellen der Requests, mit Testsplit (um die API zu testen und Kosten abzuschätzen)
#%%
test_requests = {}
test_requests_mini = {}
requests = {"batch_1":{}}
model = "llama3.1:8b-instruct-q5_K_M"
#model = "llama3.3:70b-instruct-q4_K_M"
batch_size = 1000
batch = 1

for index, row in dataset_df.iterrows():
    nl_prompt = row['nl_prompt']
    command = row['command']
    prompt = base_prompt.format(nl_prompt, command)
    request = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    if index < batch_size:
        batch_id = "batch_{}".format(batch)
        requests[batch_id][index] = request
    elif index == batch_size:
        batch += 1
        batch_size += 1000
        batch_id = "batch_{}".format(batch)
        requests[batch_id] = {}
        requests[batch_id][index] = request

print(len(test_requests))
print(len(requests))
#%%
requests["batch_1"]
#%% md
# # Synthetische Trainingsdaten mit Llama
#%%
dataset_df
#%% md
# 
#%%
errors = {}
for id, batch in tqdm(requests.items(), desc="Befrage Ollama", unit="Anfrage"):
    results = {}
    print(id)
    for index, req in batch.items():
        try:
            response = r.post("http://localhost:11434/api/generate", json=req)
            response.raise_for_status()
            result = response.json()
            results[index] = result.get("response")


        except Exception as e:
            errors[index] = str(e)
    with open(f"../data/json_results/results_{id}.json", "w") as f:
        json.dump(results, f)

print(len(errors))
#%%
results
#%% md
# Kombiniere die Resultate und ergänze sie mit dem Dataframe