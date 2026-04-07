---
title: Shad Ml2 Paper Classifier
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
license: mit
---

# Paper Classifier (arXiv topics)

Приложение принимает название статьи и/или аннотацию и возвращает наиболее вероятные тематики arXiv.  
Темы выводятся по убыванию вероятности до тех пор, пока суммарная вероятность не превысит 95%.

## Model

Приложение использует **собственную** обученную модель, сохраненную локально в папке
`ml2/week04/shad-ml2-paper-classifier/model`. Код обучения — в ноутбуке
`ml2/week04/ML2_2025_nlp_ops1.ipynb`.

## Local run

```bash
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

## Training

Откройте ноутбук `ml2/week04/ML2_2025_nlp_ops1.ipynb` и обучите модель.  
Модель сохраняется в `ml2/week04/shad-ml2-paper-classifier/model` и используется приложением.
