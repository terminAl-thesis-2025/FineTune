#%% md
# # Laden und bereinigen der Daten fürs Finetuning der Modelle
# 1. Laden der Linux/Bash Command Datasets
# 2. Laden der SQL Query Datasets
# 3. Bereinigung der Daten
# 4. Upload zu Huggingface
# 
#%% md
# Laden der benötigten Libraries
#%%
# Für diverse Datenoperationen
import pandas as pd

# Zum Laden von Datasets von Huggingface
from datasets import load_dataset, Dataset as DS
from torch.utils.data import Dataset

import sqlparse
import subprocess
#%% md
# Erstellen eines Pandas Dataframe, in dem alle Datenpunkte gemeinsam gesammelt werden
#%%
# Erstellen des leeren Dataframes
data = {
    "nl_prompt": [],
    "command": []
}

complete_data_sql = pd.DataFrame(data)
complete_data_bash = complete_data_sql.copy()

#%% md
# # Linux/Bash Commands
#%% md
# ### HF [AnishJoshi/nl2bash-custom](https://huggingface.co/datasets/AnishJoshi/nl2bash-custom)
# License=?
# 
# 
#%%
dataset1 = load_dataset("AnishJoshi/nl2bash-custom")
#%%
dataset1
#%% md
# Zusammenführung des Train/Validation/Test-Splits des Datensatzes
#%%
train_df1 = pd.DataFrame(dataset1['train'])
validation_df1 = pd.DataFrame(dataset1['validation'])
test_df1 = pd.DataFrame(dataset1['test'])

dataset1_complete = pd.concat([
    train_df1.assign(split='train'),
    validation_df1.assign(split='validation'),
    test_df1.assign(split='test')
])

dataset1_complete
#%% md
# Anhängen des Datensatzes an das Dataframe complete_data
#%%
# übernehmen der erwünschten Spalten
new_rows1 = dataset1_complete[['bash_code', 'nl_command']].copy()

# Umbenennen der übernommenen Spalten
new_rows1.rename(columns={
    'bash_code': 'command',
    'nl_command': 'nl_prompt'
}, inplace=True)

# Anhängen an complete_data
complete_data_bash = pd.concat([complete_data_bash, new_rows1], ignore_index=True)
complete_data_bash
#%% md
# ### HF [Romit2004/LinuxCommands](https://huggingface.co/datasets/Romit2004/LinuxCommands)
# License=MIT
# 
# ### Ungenügende Datenqualität! Ignorieren des Datensatzes.
#%%
#dataset2 = load_dataset("Romit2004/LinuxCommands")
#dataset2

#%% md
# ### HF [bajrangCoder/linux_cmd_alpaca](https://huggingface.co/datasets/bajrangCoder/linux_cmd_alpaca)
# License=MIT
# 
# ### Ungenügende Datenqualität! Ignorieren des Datensatzes.
#%%
#dataset3 = load_dataset("bajrangCoder/linux_cmd_alpaca")
#dataset3
#%% md
# ### HF [aelhalili/bash-commands-dataset](https://huggingface.co/datasets/aelhalili/bash-commands-dataset)
# License=MIT
#%%
dataset4 = load_dataset("aelhalili/bash-commands-dataset")
dataset4
#%% md
# Formattierung des Datensatzes in ein Dataframe
#%%
dataset4_complete = pd.DataFrame(dataset4['train'])
dataset4_complete
#%% md
# Anhängen des Datensatzes an das Dataframe complete_data
#%%
# übernehmen der erwünschten Spalten
new_rows4 = dataset4_complete[['prompt', 'response']].copy()

# Umbenennen der übernommenen Spalten
new_rows4.rename(columns={
    'response': 'command',
    'prompt': 'nl_prompt'
}, inplace=True)

# Anhängen an complete_data
complete_data_bash = pd.concat([complete_data_bash, new_rows4], ignore_index=True)
complete_data_bash
#%% md
# ### Kaggle [Kushagragoyal060705/Complex-Linux-commands-from-natual-language](https://www.kaggle.com/datasets/kushagragoyal060705/complex-linux-commands-from-natual-language)
# License=Apache 2.0
#%%
dataset5 = load_dataset("terminAl-thesis-2025/big_bash")
dataset5
#%%
dataset5 = pd.DataFrame(dataset5['train'])
dataset5
#%% md
# Anhängen des Datensatzes an das Dataframe complete_data
# 
#%%
# übernehmen der erwünschten Spalten
new_rows5 = dataset5[['input', 'output']].copy()

# Umbenennen der übernommenen Spalten
new_rows5.rename(columns={
    'output': 'command',
    'input': 'nl_prompt'
}, inplace=True)

# Anhängen an complete_data
complete_data_bash = pd.concat([complete_data_bash, new_rows5], ignore_index=True)
complete_data_bash
#%% md
# # SQL Queries/Commands
#%% md
# ### HF [zerolink/zsql-postgres-dpo](https://huggingface.co/datasets/zerolink/zsql-postgres-dpo)
# ### License=(Multiple Licenses and also "viral" Creative Commons Licenses)! <--  Ignorieren des Datensatzes.
#%%
#dataset6 = load_dataset("zerolink/zsql-postgres-dpo")
#dataset6
#%% md
# ### HF [omeryentur/text-to-postgresql](https://huggingface.co/datasets/omeryentur/text-to-postgresql)
# License=?
#%%
dataset7 = load_dataset("omeryentur/text-to-postgresql")
dataset7
#%% md
# Formattierung des Datensatzes in ein Dataframe
#%%
dataset7_complete = pd.DataFrame(dataset7['train'])
dataset7_complete
#%% md
# Anhängen des Datensatzes an das Dataframe complete_data
#%%
# übernehmen der erwünschten Spalten
new_rows7 = dataset7_complete[['question', 'query']].copy()

# Umbenennen der übernommenen Spalten
new_rows7.rename(columns={
    'query': 'command',
    'question': 'nl_prompt'
}, inplace=True)

# Anhängen an complete_data
complete_data_sql = pd.concat([complete_data_sql, new_rows7], ignore_index=True)
complete_data_sql
#%%
new_rows7
#%% md
# ### HF [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
# License=Apache-2.0
#%%
dataset8 = load_dataset("gretelai/synthetic_text_to_sql")
dataset8
#%% md
# Zusammenführung des Train/Test-Splits des Datensatzes
# 
#%%
train_df8 = pd.DataFrame(dataset8['train'])
test_df8 = pd.DataFrame(dataset8['test'])

dataset8_complete = pd.concat([
    train_df8.assign(split='train'),
    test_df8.assign(split='test')
])

dataset8_complete
#%% md
# Anhängen des Datensatzes an das Dataframe complete_data
# 
#%%
# übernehmen der erwünschten Spalten
new_rows8 = dataset8_complete[['sql_prompt', 'sql']].copy()

# Umbenennen der übernommenen Spalten
new_rows8.rename(columns={
    'sql': 'command',
    'sql_prompt': 'nl_prompt'
}, inplace=True)

# Anhängen an complete_data
complete_data_sql = pd.concat([complete_data_sql, new_rows8], ignore_index=True)
complete_data_sql
#%% md
# # Bereinigung der Daten
#%% md
# Entferne Duplikate und Prüfe SQL-Commands auf Validität
#%%
# Filtere nach eindeutigen Werten
unique_sql_nl_prompts = complete_data_sql["nl_prompt"].unique()
unique_sql_commands = complete_data_sql["command"].unique()

print(f"Length of original DataFrame: {len(complete_data_sql)}")
print(f"Unique NL Prompts: {len(unique_sql_nl_prompts)}")
print(f"Unique Commands: {len(unique_sql_commands)}")

# Entferne doppelte oder mehrfache Prompts (da mehr Commands als Prompts, ansonsten doppelte Commands entfernen)
complete_data_sql = complete_data_sql.drop_duplicates(subset="nl_prompt", keep="first").copy()
print(f"Length of complete data: {len(complete_data_sql)}")
print(f"Unique Commands: {len(complete_data_sql['command'].unique())}")
print(f"Verbleibende doppelte Einträge: {len(complete_data_sql)-len(complete_data_sql['command'].unique())}")

def is_valid_sql(command: str) -> bool:
    try:
        parsed = sqlparse.parse(command)
        return bool(parsed) and all(stmt.tokens for stmt in parsed)
    except Exception:
        return False

# Apply syntax check to each row
complete_data_sql["check"] = complete_data_sql["command"].apply(is_valid_sql)
#%% md
# Prüfe Bash-Commands auf Validität
# 
# **Vorsicht! Sicherheitshalber diesen Befehl in einer gesicherten Umgebung (Docker oder VM) ausführen**
#%%
# Filtere nach eindeutigen Werten
unique_bash_nl_prompts = complete_data_bash["nl_prompt"].unique()
unique_bash_commands = complete_data_bash["command"].unique()

print(f"Length of original DataFrame: {len(complete_data_bash)}")
print(f"Unique NL Prompts: {len(unique_bash_nl_prompts)}")
print(f"Unique Commands: {len(unique_bash_commands)}")

# Entferne doppelte oder mehrfache Prompts (da mehr Commands als Prompts, ansonsten doppelte Commands entfernen)
complete_data_bash = complete_data_bash.drop_duplicates(subset="nl_prompt", keep="first").copy()
print(f"Length of complete data: {len(complete_data_bash)}")
print(f"Unique Commands: {len(complete_data_bash['command'].unique())}")
print(f"Verbleibende doppelte Einträge: {len(complete_data_bash)-len(complete_data_bash['command'].unique())}")

def is_valid_bash_syntax(command: str) -> bool:
    try:
        result = subprocess.run(
            ['bash', '-n'],
            input=command.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False

complete_data_bash["check"] = complete_data_bash["command"].apply(is_valid_bash_syntax)

#%% md
# Zusammenführen der Dataframes
#%%
complete_data = pd.concat([complete_data_sql, complete_data_bash])
#%% md
# Prüfe auf falsche Syntax und entferne diese
#%%
false_count = (complete_data['check'] == False).sum()
print(f"Einträge mit falscher Syntax zu entfernen: {false_count}")

# Remove rows where column 'A' is False
complete_data = complete_data[complete_data['check']]
#%%
complete_data
#%% md
#  # Upload zu Huggingface
#%%
hf_complete_data = DS.from_pandas(complete_data)
hf_complete_data.push_to_hub("terminAl-thesis-2025/combined_dataset")
#%%
