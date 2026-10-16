# Emotion Risk Engine V2

Proyecto experimental y educativo para clasificar texto en español en cuatro niveles de riesgo emocional. Incluye un predictor basado en BETO, una API FastAPI y persistencia en PostgreSQL. La demo web está en [AVARI’S//LAB](https://avaris-lab.vercel.app/).

> **Aviso:** Emotion Risk Engine no es un dispositivo médico ni un sistema clínico validado. Sus resultados son estimaciones del modelo; no sustituyen la evaluación, el diagnóstico ni el criterio de un profesional de la salud. Una confianza alta no garantiza que una predicción sea correcta ni demuestra precisión clínica.
>
> La API guarda en la base de datos el texto normalizado por el predictor y escribe en sus logs el texto original recibido. No introduzcas en la demo pública datos personales, información clínica identificable ni otros textos sensibles. El código actual no implementa anonimización, consentimiento ni una política de retención.

## Arquitectura actual

```text
Navegador (AVARI’S//LAB)
        │ HTTPS: /health y /predict
        ▼
Cloudflare Quick Tunnel (opcional en Colab)
        │
        ▼
FastAPI ───────► BETO desde Hugging Face Hub
   │
   └───────────► PostgreSQL (por ejemplo, Neon)
                 guarda texto normalizado, clase,
                 confianza y fecha
```

El frontend usa `VITE_API_URL` como URL base del backend. La API permite el origen de producción `https://avaris-lab.vercel.app` y los orígenes locales configurados en `app/api/main.py`. CORS permite que el navegador lea respuestas; no autentica clientes ni protege endpoints.

## Modelo y etiquetas

El predictor de la API, `app/services/predictor.py`, carga `AutoTokenizer` y `AutoModelForSequenceClassification` desde `MODEL_DIR`. La configuración (`app/utils/config.py`) usa un directorio local `models/emotion-risk-beto` si existe; en caso contrario usa el repositorio `avarixo/emotion-risk-beto` de Hugging Face Hub. El identificador BETO base `dccuchile/bert-base-spanish-wwm-cased` también aparece en la configuración, pero no es el valor que el predictor carga por defecto.

Los índices y nombres definidos por la API son:

| Índice (`label`) | Clase devuelta |
| ---: | --- |
| 0 | Sin riesgo |
| 1 | Riesgo bajo |
| 2 | Riesgo moderado |
| 3 | Riesgo alto |

El predictor normaliza espacios, mayúscula inicial y puntuación final, y limita la entrada del modelo a 128 tokens. `POST /predict` acepta textos de 1 a 1000 caracteres.

## API disponible

FastAPI genera documentación interactiva en `/docs` cuando el servicio está en ejecución.

| Método y ruta | Comportamiento |
| --- | --- |
| `GET /` | Nombre y versión de la API. |
| `GET /health` | Devuelve `{"status":"ok"}`; es una respuesta básica de estado. |
| `GET /version` | Devuelve la versión. |
| `POST /predict` | Recibe `{"text":"..."}` y devuelve la predicción. |
| `GET /history` | Devuelve los registros, incluido su texto. |
| `GET /history/{prediction_id}` | Devuelve un registro, incluido su texto. |
| `PATCH /history/{prediction_id}`, `PUT /history/{prediction_id}` | Actualiza el campo `risk`. |
| `DELETE /history/{prediction_id}` | Elimina un registro. |
| `GET /stats` | Devuelve cantidad total, confianza promedio y distribución por clase. |
| `GET /alerts?risk_threshold=3` | Devuelve registros desde el umbral indicado, incluido su texto. El valor predeterminado es 3. |
| `GET /export/json`, `GET /export/csv` | Exporta los registros, incluidos sus textos. |

La respuesta de `POST /predict` tiene los campos `label` (entero de 0 a 3), `class` (nombre de la clase), `confidence` (confianza para la clase elegida) y `probabilities` (probabilidad por nombre de clase). Es un resultado del modelo, no una recomendación ni diagnóstico.

### Persistencia y acceso

La tabla `predictions` almacena `text`, `risk` (índice de clase), `confidence` y `created_at`; el identificador es generado por la base de datos. El texto guardado es el texto normalizado por el predictor. Además, el endpoint registra en el log el texto original recibido, junto con clase y confianza. Si falla el registro en la base, el predictor hace rollback e informa una advertencia; el flujo de inferencia puede continuar.

Las rutas de historial, alertas y exportación devuelven textos. En las rutas inspeccionadas no hay autenticación ni autorización. No expongas esta API con datos reales o sensibles ni asumas que CORS restringe el acceso a otros clientes.

## Requisitos e instalación

Se requiere Python 3.10 o posterior, acceso a una base PostgreSQL y acceso al modelo local o a Hugging Face Hub. PyTorch y Transformers pueden descargar paquetes grandes; el modelo se descarga desde Hub al cargarlo si no existe localmente.

Desde la raíz del repositorio, instala las dependencias de API y las dos dependencias ML que `requirements.txt` no incluye:

```bash
python -m pip install -r requirements.txt
python -m pip install torch transformers
```

`pyproject.toml` declara además dependencias de ML y análisis de datos, pero el Dockerfile instala únicamente `requirements.txt`. No se afirma que la imagen Docker actual pueda iniciar: le faltan `torch` y `transformers`, que importa el predictor. También `.dockerignore` no excluye `.env`, mientras que el Dockerfile copia el contexto completo. Revisa y corrige esas limitaciones antes de construir o distribuir una imagen con archivos locales.

## Ejecución local

Configura `DATABASE_URL` en el entorno del proceso con una base PostgreSQL a la que tengas acceso. No pegues la URL en documentación, capturas ni logs compartidos. El valor de ejemplo de `.env.example` corresponde al servicio `postgres` de Docker Compose y no es una configuración para Neon ni para una ejecución local fuera de Docker.

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Al importar la aplicación se llama a `Base.metadata.create_all`; esto crea tablas ausentes, pero no sustituye un sistema de migraciones. El predictor se instancia al cargar las rutas, así que el arranque requiere que el modelo sea accesible.

## Docker Compose

El repositorio incluye `Dockerfile` y `docker-compose.yml`. Compose espera un archivo `.env` para el servicio API y configura un PostgreSQL local con volumen persistente. No uses las credenciales de desarrollo de Compose para servicios expuestos. La imagen tiene la limitación de dependencias indicada arriba; por eso estos archivos describen configuración existente, no una ejecución limpia verificada.

```bash
docker compose up --build
```

## Neon y Google Colab

El notebook documenta cómo configurar `DATABASE_URL` mediante Colab Secrets, convertir el controlador a `psycopg2` y probar conectividad con `SELECT 1`. Esa comprobación no escribe datos.

Guía interactiva: [notebooks/emotion-risk-engine-v4.1.ipynb](notebooks/emotion-risk-engine-v4.1.ipynb).

La celda de predicción directa del notebook no pasa una sesión de base de datos; la prueba HTTP a `POST /predict` sí crea un registro en la base configurada. Las solicitudes a ese endpoint desde cualquier cliente también pueden crear registros. Usa una base de prueba y únicamente texto ficticio. El notebook conserva outputs de una sesión anterior, incluida una URL temporal del túnel: no la reutilices y limpia los outputs antes de compartir una copia. Esos outputs no garantizan que Colab o la API estén disponibles actualmente.

## Cloudflare Quick Tunnel y frontend

El notebook puede iniciar `cloudflared tunnel --url http://localhost:8000` y detectar la URL pública `https://…trycloudflare.com`. Un Quick Tunnel proporciona una URL temporal que puede cambiar al crear otra sesión; no debe guardarse como dirección permanente.

Para conectar el frontend, configura `VITE_API_URL` con la URL base vigente del backend, sin añadir `/health` ni `/predict`. En Vercel, actualiza la variable para el entorno de producción y vuelve a desplegar para que Vite incorpore el valor en la compilación. Comprueba que el backend esté activo y permita el origen del sitio. El frontend no descubre automáticamente una nueva URL temporal de Colab.

## Métricas disponibles

El reporte versionado en [`reports/classification_report.txt`](reports/classification_report.txt) informa resultados sobre 782 ejemplos del conjunto de evaluación: accuracy 0.9757 y macro F1 0.9760. Por clase, el soporte es 197, 200, 200 y 185 para las etiquetas 0–3, respectivamente. La [matriz de confusión](reports/confusion_matrix.png) también está versionada.

Son métricas del reporte disponible en este repositorio; no las he recalculado en esta revisión. Describen ese conjunto y configuración, no garantizan rendimiento en otros textos, poblaciones o contextos. El recall 1.0000 reportado para la clase 3 no demuestra que el sistema vaya a detectar todos los casos reales ni valida uso clínico.

## Estructura relevante

```text
app/
├── api/             # Aplicación FastAPI, rutas y esquemas
├── database/        # SQLAlchemy, conexión y modelo Prediction
├── engine/          # Módulos de análisis/reglas
├── preprocessing/   # Limpieza de datos
├── services/        # Predictor BETO y modelo anterior
└── utils/           # Configuración, logging y normalización
data/                # Datos del proyecto
docs/                # Diseño, anotación y planificación
notebooks/           # Guías interactivas de Colab
reports/             # Reportes y visualizaciones versionados
scripts/             # Preparación, entrenamiento y evaluación
tests/               # Comprobaciones existentes
```

## Trabajo futuro (no implementado)

- **V3 orientada a investigación:** evaluar datos autorizados, consentimiento informado cuando corresponda, minimización de datos, revisión de etiquetas, control de calidad y evaluación rigurosa antes de entrenar una nueva versión.
- **Apoyo a profesionales de la salud:** explorar si la herramienta puede apoyar la revisión de señales, sin sustituir a profesionales ni afirmar detección o diagnóstico de enfermedades crónicas o psicológicas sin validación específica.
- **Plataformas digitales:** investigar posibles herramientas para identificar señales en mensajes, considerando privacidad, permisos, falsos positivos y negativos, y protocolos de actuación adecuados.

Estas ideas son líneas de investigación, no funcionalidades disponibles ni promesas comerciales.

## Licencia y estado de validación

Este README documenta el código y los artefactos presentes en el repositorio. No implica que se haya ejecutado aquí una compilación Docker, una evaluación nueva, pruebas de integración o una validación clínica.
