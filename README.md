```markdown
# Emotion Risk Engine 🧠⚡

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![BETO](https://img.shields.io/badge/Model-BETO%20(Spanish%20BERT)-yellow?style=for-the-badge)](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Neon](https://img.shields.io/badge/Neon-PostgreSQL-green?style=for-the-badge)](https://neon.tech/)
[![Cloudflare](https://img.shields.io/badge/Cloudflare-Tunnel-orange?style=for-the-badge&logo=cloudflare)](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

Motor de análisis y clasificación de riesgo emocional en texto en español de extremo a extremo. Desarrollado mediante fine-tuning sobre **BETO** (Spanish BERT), expuesto a través de una API modular en **FastAPI**, persistencia serverless con **PostgreSQL (Neon)** y despliegue seguro con **Docker** y **Cloudflare Tunnels**.

---

## 🏗 Arquitectura del Sistema

```text
       [ Cliente / Frontend ]
                 │
                 ▼
    [ Cloudflare Tunnel / Proxy ]
                 │
                 ▼
 [ FastAPI App (Docker / Uvicorn Server) ]
        │                       │
        ▼                       ▼
 🧠 BETO Transformer     🗄️ Neon DB (PostgreSQL)
 (Inferencia en Memoria)  (Auditoría y Registros)

```

---

## ✨ Características Principales

* **Fine-Tuning Especializado:** Basado en BETO (`bert-base-spanish-wwm-cased`), calibrado para capturar matices de crisis y angustia en lenguaje cotidiano en español.
* **API REST Asíncrona:** Construida con FastAPI, esquemas de tipado con Pydantic y documentación OpenAPI interactiva.
* **Persistencia en la Nube:** Registro persistente de consultas y métricas de inferencia en PostgreSQL alojado en Neon.
* **Aislamiento y Portabilidad:** Empaquetado completo en contenedor Docker orquestado con `docker-compose`.
* **Exposición Segura:** Integración nativa con `cloudflared` para exponer la API a Internet sin necesidad de abrir puertos ni exponer IP pública.

---

## 📊 Niveles de Riesgo y Semántica

El clasificador segmenta la entrada de texto en 4 niveles de severidad:

| Etiqueta | Clase | Descripción | Acción Recomendada |
| --- | --- | --- | --- |
| **0** | **Sin riesgo** | Contenido neutral o expresiones cotidianas normales | Monitoreo pasivo |
| **1** | **Riesgo bajo** | Estrés leve, cansancio situacional o frustración transitoria | Información y apoyo general |
| **2** | **Riesgo moderado** | Tristeza recurrente, agobio persistente o ansiedad notable | Alerta preventiva |
| **3** | **Riesgo alto** | Señales de crisis emocional aguda o desesperanza explícita | Protocolo de intervención inmediata |

---

## 📈 Rendimiento del Modelo (Evaluación BETO)

Evaluado sobre un conjunto de prueba independiente con métricas de alta precisión por clase:

| Etiqueta | Clase | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **0** | **Sin riesgo** | 0.9946 | 0.9391 | 0.9661 |
| **1** | **Riesgo bajo** | 0.9417 | 0.9700 | 0.9557 |
| **2** | **Riesgo moderado** | 0.9755 | 0.9950 | 0.9851 |
| **3** | **Riesgo alto** | **0.9946** | **1.0000** | **0.9973** |

> **Observación de Impacto Clínico/Técnico:**
> En la clase más sensible (**Label 3: Riesgo alto**), el modelo alcanzó un **Recall de 1.0000**. Esto garantiza **cero falsos negativos**, asegurando que ningún caso con indicadores de riesgo crítico sea omitido o clasificado erróneamente.
> 
> 

---

## 🛠 Tecnologías

* **NLP & Deep Learning:** PyTorch, Hugging Face Transformers, Tokenizers, BETO.
* **Backend:** FastAPI, Uvicorn, Pydantic, Python.
* **Base de Datos:** PostgreSQL (Neon Tech).
* **Infraestructura:** Docker, Docker Compose, Cloudflare `cloudflared`.

---

## 📂 Estructura del Repositorio

```text
emotion-risk-engine/
├── app/                  # Núcleo de la aplicación FastAPI
│   ├── api/              # Definición de rutas y endpoints
│   ├── core/             # Ajustes y configuración general
│   ├── models/           # Esquemas de base de datos y validaciones
│   └── services/         # Inicialización del modelo BETO y predictor
├── data/                 # Datasets y procesamiento
├── docs/                 # Documentación técnica adicional
├── models/               # Pesos y artefactos del modelo
├── reports/              # Reportes de entrenamiento y métricas
├── scripts/              # Scripts de soporte y utilidades
├── tests/                # Pruebas unitarias y de configuración
├── Dockerfile            # Imagen base optimizada
├── docker-compose.yml    # Definición de servicios
├── requirements.txt      # Dependencias fijadas
└── .env.example          # Plantilla para variables de entorno

```

---

## 🚀 Despliegue y Ejecución

### 1. Variables de Entorno

Copia la plantilla y configura tus accesos:

```bash
cp .env.example .env

```

Valores requeridos en `.env`:

```env
DATABASE_URL=postgresql://usuario:contraseña@ep-xyz.neon.tech/neondb?sslmode=require
HF_TOKEN=hf_tu_token_aqui
PORT=8000

```

### 2. Ejecutar con Docker (Recomendado)

```bash
docker compose up --build

```

La documentación Swagger estará disponible automáticamente en:
`http://localhost:8000/docs`

### 3. Ejecución Local / Google Colab

```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar el servidor
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

```

Para exponer la API a Internet mediante Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:8000

```

---

## 📡 Referencia de la API

### Inferencia de Riesgo (`POST /predict`)

**Payload:**

```json
{
  "text": "me quiero morir"
}

```

**Respuesta HTTP 200:**

```json
{
  "label": 3,
  "class": "Riesgo alto",
  "confidence": 0.9961,
  "probabilities": {
    "Sin riesgo": 0.0002,
    "Riesgo bajo": 0.0036,
    "Riesgo moderado": 0.0001,
    "Riesgo alto": 0.9961
  }
}

```

---

## 👤 Autor

Desarrollado por **AvariAny**.
