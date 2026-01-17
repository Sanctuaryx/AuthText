# Detector de Texto IA vs Humano (TFM)

Sistema completo para **clasificar textos en español** como **IA (1)** vs **Humano (0)** y compararlos de forma homogénea, incluyendo **entrenamiento**, **evaluación** y **exposición vía API REST**.

Modelos incluidos:

1. **BiLSTM (embeddings aleatorios)** — entrenado desde cero (baseline).
2. **Word2Vec + BiLSTM** — embeddings estáticos entrenados en tu corpus + BiLSTM.
3. **BERT fine-tuned** (dccuchile/bert-base-spanish-wwm-cased) — fine-tuning supervisado con *chunking/stride* y agregación por documento.

---

## Índice

- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Datos](#datos)
- [Entrenamiento](#entrenamiento)
  - [1) BiLSTM baseline](#1-bilstm-baseline)
  - [2) Word2Vec + BiLSTM](#2-word2vec--bilstm)
  - [3) BERT fine-tuned](#3-bert-fine-tuned)
- [Evaluación y comparación](#evaluación-y-comparación)
- [API REST](#api-rest)
- [Artefactos esperados](#artefactos-esperados)
- [Troubleshooting](#troubleshooting)

---

## Estructura del repositorio

> Puede variar ligeramente según cómo lo tengas organizado, pero el flujo esperado es:

```
.
├─ data/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test_es.csv
├─ logic/
│  ├─ models/
│  ├─ training/
│  │  ├─ train_bert.py
│  │  ├─ train_w2v.py
│  │  └─ (script de entrenamiento BiLSTM, p. ej. train.py o train_bilstm.py)
│  └─ artifacts/
│     ├─ bilstm_rand/
│     ├─ bilstm_w2v/
│     └─ bert/
├─ tests/
│  └─ eval_test.py
├─ api.py
├─ requirements.txt
└─ README.md
```

---

## Requisitos

- **Python 3.10+** recomendado (3.11/3.12 también funciona).
- Entorno virtual (`venv`).
- **PyTorch** con CUDA (opcional pero recomendado si tienes GPU).

Instalación (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

> **GPU**: si `torch.cuda.is_available()` te da `False`, normalmente es porque instalaste una build CPU.
> Instala PyTorch CUDA desde la web oficial de PyTorch para tu versión de CUDA.

---

## Datos

Los CSV deben tener al menos:

- `text` (string)
- `generated` (0 = humano, 1 = IA)

Rutas típicas:

- `data/train.csv`
- `data/val.csv`
- `data/test_es.csv`

---

## Entrenamiento

> **Importante**: ejecuta los comandos **desde la raíz del proyecto**.

### 1) BiLSTM baseline

Salida recomendada:

- `logic/artifacts/bilstm_rand`

Comando (si tu script está en raíz como `train.py`):

```powershell
python -u train.py --train_path data/train.csv --val_path data/val.csv --out_dir logic/artifacts/bilstm_rand
```

Si lo tienes dentro de `logic/training/`:

```powershell
python -u logic/training/train.py --train_path data/train.csv --val_path data/val.csv --out_dir logic/artifacts/bilstm_rand
```

Artefactos:

```
logic/artifacts/bilstm_rand/
  model.pt
  vocab.json
  config.json
```

---

### 2) Word2Vec + BiLSTM

Salida recomendada:

- `logic/artifacts/bilstm_w2v`

#### 2.1 Entrenar Word2Vec

```powershell
python -u logic/training/train_w2v.py --train_path data/train.csv --out_dir logic/artifacts/bilstm_w2v/embeddings
```

Esperado:

```
logic/artifacts/bilstm_w2v/embeddings/
  bilstm_w2v.model
  bilstm_w2v.model.syn1neg.npy
  bilstm_w2v.model.wv.vectors.npy
```

#### 2.2 Entrenar BiLSTM con Word2Vec

```powershell
python -u train.py `
  --train_path data/train.csv `
  --val_path data/val.csv `
  --w2v_path logic/artifacts/bilstm_w2v/embeddings/bilstm_w2v.model `
  --out_dir logic/artifacts/bilstm_w2v
```

Artefactos:

```
logic/artifacts/bilstm_w2v/
  model.pt
  vocab.json
  config.json
  embeddings/   (si lo conservas)
```

---

### 3) BERT fine-tuned

Aquí **NO** hay `model.pt`. Se guarda en formato HuggingFace con **`model.safetensors`** + tokenizer + `config.json`.

Salida recomendada:

- `logic/artifacts/bert`

Comando típico:

```powershell
python -u logic/training/train_bert.py `
  --train_csv data/train.csv `
  --val_csv data/val.csv `
  --test_csv data/test_es.csv `
  --out_dir logic/artifacts/bert `
  --max_length 384 `
  --stride 128 `
  --agg median `
  --epochs 3 `
  --batch_size 8 `
  --grad_accum 2 `
  --lr 2e-5
```

Artefactos mínimos:

```
logic/artifacts/bert/
  config.json
  model.safetensors
  tokenizer_config.json / special_tokens_map.json / vocab.txt / tokenizer.json (según tokenizer)
  test_metrics.json (si evalúas en test)
  calibrator.joblib (si tu pipeline calibra; opcional)
```

---

## Evaluación y comparación

```powershell
python -u tests/eval_test.py `
  --test_path data/test_es.csv `
  --bilstm_rand_dir logic/artifacts/bilstm_rand `
  --bilstm_w2v_dir logic/artifacts/bilstm_w2v `
  --bert_dir logic/artifacts/bert
```

Genera:
- `logic/artifacts/<modelo>/test_metrics.json`
- `metrics/all_models_metrics.json`

---

## API REST

Arranque:

```powershell
python -u api.py
```

Endpoints:
- `GET /health`
- `GET /models`
- `POST /predict` (body: `{ "text": "...", "model": "bert|bilstm_rand|bilstm_w2v" }`)
- `POST /predict/<model_name>`

PowerShell (UTF-8):

```powershell
$payload = @{ text="Texto largo en español para evaluar..." ; model="bert" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

curl:

```bash
curl -X POST http://localhost:8001/predict   -H "Content-Type: application/json"   -d '{"text":"Texto largo en español para evaluar...","model":"bert"}'
```

---

## Artefactos esperados

- **BiLSTM**: `model.pt`, `vocab.json`, `config.json`
- **BERT fine-tuned**: `model.safetensors` + tokenizer files + `config.json`

---

## Troubleshooting

- `ModuleNotFoundError: logic`: ejecuta desde la raíz y/o exporta `PYTHONPATH`.
- PowerShell + acentos: envía el body como bytes UTF-8.
- `PermissionError`: evita rutas con OneDrive bloqueando y comprueba permisos.
- GPU no detectada: instala PyTorch CUDA.
