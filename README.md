# IA Text Detector (iatd)

Detector de texto generado por IA vs. humano en castellano.

Incluye:
- *Featurizer* (embeddings + rasgos estilométricos + perplejidad)
- Modelo baseline (SVM calibrado)
- Modelo Transformer (BETO/roberta-es) para fine-tuning
- API REST con Flask
- UI de demo con Streamlit

## Instalación rápida

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

python -m spacy download es_core_news_md
