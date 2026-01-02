# Detector de Texto IA vs Humano (TFM)

Proyecto para clasificación binaria **IA (1)** vs **Humano (0)** en textos en español usando:

- **BiLSTM + Atención** (modelo entrenado desde cero)
- **Word2Vec + BiLSTM** (embeddings estáticos entrenados en el corpus)
- **BERT (embeddings) + MLP** (embeddings contextuales + clasificador ligero)

Incluye una **API REST** para inferencia con los 3 modelos.

---

## Estructura esperada

> Los nombres de paquetes pueden ser `logic/` o `iatd/` (según tu refactor).  
> **Lo importante:** el código está en el paquete (logic/iatd) y los modelos entrenados en `artifacts/`.

```
.
├── logic/  (o iatd/)
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── custom_bilstm.py
│       ├── dataset.py
│       └── vocab.py
├── src/
│   ├── bert_embeddings.py
│   └── train_bert.py
├── tests/
│   └── eval_test.py
├── artifacts/                # <-- modelos entrenados (se generan)
├── data/                     # <-- datasets
├── train.py                  # <-- entrena BiLSTM (rand / w2v)
└── api.py                    # <-- API Flask
```

> **Importante:** los modelos entrenados NO se guardan dentro del paquete (`logic/` o `iatd/`).  
> Todo va en `artifacts/`.

---

## Requisitos

- Python 3.10+ recomendado
- PyTorch (GPU opcional)
- `transformers` + `safetensors` para BERT
- `gensim` para Word2Vec
- `flask` para la API

Instalación típica (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

---

## Datos (formato)

Los CSV deben tener al menos estas columnas:

- `text` (string)
- `generated` (0 = humano, 1 = IA)

Ejemplos:
- `data/train.csv`
- `data/val.csv`
- `data/test_es.csv`

---

# Entrenamiento de los 3 modelos

## 1) BiLSTM base (embeddings aleatorios) → `artifacts/bilstm_rand`

Entrenar:

```powershell
python train.py --train_path data/train.csv --val_path data/val.csv --out_dir artifacts/bilstm_rand
```

Artefactos generados:

```
artifacts/bilstm_rand/
  model.pt
  vocab.json
  config.json
```

Evaluar en test:

```powershell
python tests/eval_test.py --test_path data/test_es.csv --bilstm_rand_dir artifacts/bilstm_rand
```

---

## 2) Word2Vec + BiLSTM → `artifacts/bilstm_w2v`

### 2.1 Entrenar Word2Vec (embeddings) → `artifacts/w2v.model`

> Esto entrena **solo** Word2Vec (sin labels). **No es el clasificador.**  
> Usa el script que exista en tu repo (uno de estos dos):
>
> - `python iatd/training/train_w2v.py ...`  
> - `python src/train_w2v.py ...`

Comando (opción A):

```powershell
python iatd/training/train_w2v.py --train_path data/train.csv --out_path artifacts/w2v.model
```

Comando (opción B):

```powershell
python src/train_w2v.py --train_path data/train.csv --out_path artifacts/w2v.model
```

Genera:

- `artifacts/w2v.model`
- `artifacts/w2v.model.syn1neg.npy`
- `artifacts/w2v.model.wv.vectors.npy`

### 2.2 Entrenar BiLSTM usando embeddings W2V

> `train.py` debe aceptar `--w2v_path` y usarlo para inicializar la capa `nn.Embedding`.

```powershell
python train.py --train_path data/train.csv --val_path data/val.csv --w2v_path artifacts/w2v.model --out_dir artifacts/bilstm_w2v
```

Artefactos generados:

```
artifacts/bilstm_w2v/
  model.pt
  vocab.json
  config.json
```

Evaluar en test:

```powershell
python tests/eval_test.py --test_path data/test_es.csv --bilstm_w2v_dir artifacts/bilstm_w2v
```

---

## 3) BERT embeddings + MLP → `artifacts/bert`

### 3.1 Generar embeddings BERT (train / val / test)

```powershell
python src/bert_embeddings.py --input data/train.csv   --output data/bert/bert_train.npz
python src/bert_embeddings.py --input data/val.csv     --output data/bert/bert_val.npz
python src/bert_embeddings.py --input data/test_es.csv --output data/bert/bert_test.npz
```

Cada `.npz` contiene:
- `X`: embeddings (N, 768)
- `y`: etiquetas (N,)

> Nota: para evitar problemas de seguridad con `.bin` y torch < 2.6, el script debe cargar con `use_safetensors=True`.

### 3.2 Entrenar el MLP sobre embeddings

```powershell
python src/train_bert.py --train_npz data/bert/bert_train.npz --val_npz data/bert/bert_val.npz --test_npz data/bert/bert_test.npz --out_dir artifacts/bert
```

Artefactos generados:

```
artifacts/bert/
  model.pt
  config.json
  test_metrics.json
```

Evaluación (opcional, también se puede hacer desde el comparador global):

```powershell
python tests/eval_test.py --bert_mlp_dir artifacts/bert --bert_test_npz data/bert/bert_test.npz
```

---

# Comparar los 3 modelos (mismo test)

```powershell
python tests/eval_test.py `
  --test_path data/test_es.csv `
  --bilstm_rand_dir artifacts/bilstm_rand `
  --bilstm_w2v_dir artifacts/bilstm_w2v `
  --bert_mlp_dir artifacts/bert `
  --bert_test_npz data/bert/bert_test.npz
```

Esto imprime una tabla comparativa y guarda:

```
artifacts/all_models_metrics.json
```

---

# API REST (Flask)

La API permite inferencia con:
- `bilstm_rand`
- `bilstm_w2v`
- `bert`

## 1) Arrancar la API

```powershell
python api.py
```

Por defecto escucha en:

- `http://localhost:8001`

## 2) Ver estado y modelos cargados

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET
```

Lista de modelos disponibles:

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/models" -Method GET
```

## 3) Petición de predicción (PowerShell)

> Nota: para evitar problemas de codificación con `ñ`, se envía el body como bytes UTF-8.

### a) BiLSTM rand

```powershell
$payload = @{ text="Texto largo en español para evaluar..." ; model="bilstm_rand" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

### b) BiLSTM + Word2Vec

```powershell
$payload = @{ text="Texto largo en español para evaluar..." ; model="bilstm_w2v" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

### c) BERT + MLP

```powershell
$payload = @{ text="Texto largo en español para evaluar..." ; model="bert" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

## 4) Petición con curl (Linux/macOS)

```bash
curl -X POST http://localhost:8001/predict   -H "Content-Type: application/json"   -d '{"text":"Texto largo en español para evaluar...","model":"bert"}'
```

## 5) Respuesta esperada (ejemplo)

```json
{
  "model": "bilstm_rand",
  "score": 0.73,
  "decision": "IA",
  "threshold": 0.30,
  "confidence": "high",
  "text_length": 120,
  "min_words": 30
}
```

---

## Variables de entorno útiles (opcional)

Puedes cambiar rutas y modelo por defecto sin tocar código:

```powershell
$env:BILSTM_RAND_DIR="artifacts/bilstm_rand"
$env:BILSTM_W2V_DIR="artifacts/bilstm_w2v"
$env:BERT_DIR="artifacts/bert"
$env:BERT_BASE_MODEL="dccuchile/bert-base-spanish-wwm-cased"
$env:DEFAULT_MODEL="bilstm_rand"
python api.py
```

---

## Checklist de artefactos (para que la API cargue)

Antes de arrancar la API, deben existir:

**BiLSTM rand**
- `artifacts/bilstm_rand/model.pt`
- `artifacts/bilstm_rand/vocab.json`
- `artifacts/bilstm_rand/config.json`

**BiLSTM + W2V**
- `artifacts/bilstm_w2v/model.pt`
- `artifacts/bilstm_w2v/vocab.json`
- `artifacts/bilstm_w2v/config.json`

**BERT + MLP**
- `artifacts/bert/model.pt`
- `artifacts/bert/config.json`

> Para BERT+MLP, la API calcula embeddings al vuelo usando el modelo base definido por `BERT_BASE_MODEL`.
