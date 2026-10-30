# Safety Incident Prediction & Root Cause Analysis Platform

An enterprise-grade, machine-learning and NLP-powered safety management platform designed for the petrochemical, chemical, and manufacturing industries. The system analyzes raw incident reports, extracts key entities using domain-specific NLP, and predicts **Risk Level**, **Severity Level**, and **Root Cause Category** using multi-target CatBoost classification pipelines.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Environment Variables](#-environment-variables)
- [Database Setup & Migrations](#-database-setup--migrations)
- [Running the Application](#-running-the-application)
- [ML & NLP Pipeline Explanation](#-ml--nlp-pipeline-explanation)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Deployment Options](#-deployment-options)
- [Screenshots & UI Walkthrough](#-screenshots--ui-walkthrough)
- [License Recommendation](#-license-recommendation)

---

## 🚀 Overview

Workplace safety incidents in chemical plants and manufacturing facilities require rapid, structured triage and objective causal analysis. Traditional manual reporting often suffers from incomplete categorical tags, subjective severity ratings, and delays in identifying systemic hazards.

This platform bridges the gap between unstructured field narratives and auditable compliance data:
1. **Automated NLP Feature Extraction**: Parses free-form text narratives to identify chemical substances, equipment tags, injury types, locations, and contributing actions.
2. **Multi-Target Predictive ML**: Simultaneously infers risk levels, OSHA/internal severity tiers, and root cause categories using calibrated gradient boosting models.
3. **Audited Data Architecture**: Maintains strict physical and query isolation between historical baseline datasets (8,000 records) and live user-submitted incident reports in PostgreSQL.
4. **Interactive Safety Analytics**: Delivers production-grade KPI executive dashboards, trend analytics, risk matrices, and report management interfaces.

---

## ✨ Key Features

- **End-to-End Incident Workflow**:
  - Direct narrative submission with automated domain entity extraction.
  - Interactive verification interface enabling safety officers to correct or augment extracted tags.
  - Multi-target inference delivering predicted classes, prediction probabilities, and confidence scores.
- **Document Attachment & Audit Storage**:
  - Secure document upload subsystem with SHA-256 integrity hashing and unique disk UUID decoupling.
  - Strict privacy protection with local/S3-compatible persistent storage.
- **Strict Data Integrity & Partitioning**:
  - Isolated database storage distinguishing historical training data from live operational submissions.
  - Dedicated PostgreSQL partition views (`v_historical_incidents`, `v_user_incidents`).
  - Zero contamination guarantee: Historical records never contain synthetic model predictions; user submissions never carry unverified ground-truth labels.
- **Robust Model Diagnostics & Fallbacks**:
  - Hybrid NLP fallback engine combining rule-based spaCy NER with Transformer semantic embeddings (`all-MiniLM-L6-v2`).
  - Graceful degradation when offline or when encountering unknown entities.

---

## 🏗 Architecture & Tech Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│ Streamlit Frontend  (safety_incident_ui/app.py)  — Port 8501           │
│  • Executive KPI Dashboards & Real-time Risk Matrices                  │
│  • Incident Reporting Form with Live NLP Parsing Preview               │
│  • Historical Dataset Explorer & Comparative Analytics                │
│  • Offline / Mock Fallback Mode for Demonstrations                     │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ HTTP / JSON REST APIs
┌──────────────────────────────▼─────────────────────────────────────────┐
│ FastAPI Backend  (app.py)  — Port 8502                                 │
│  • POST /predict           • POST /upload        • GET /health         │
│  • GET  /incidents         • GET  /incidents/{id} • GET /analytics/kpis│
│  • GET  /models            • GET  /nlp/extract   • GET /evaluation/... │
│  • Joblib Inference Pipeline Wrapper (CatBoost MultiClass Classifiers) │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ SQLAlchemy 2.0 ORM (psycopg driver)
┌──────────────────────────────▼─────────────────────────────────────────┐
│ PostgreSQL 14+ Relational Database  (safety_incident_db)               │
│  • incidents               • prediction_runs     • predictions         │
│  • analysis_results        • incident_files      • model_versions      │
│  • dataset_import_logs     • v_historical_incidents (View)             │
│  • v_user_incidents (View) • users                                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Backend**: FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, Psycopg (v3), Alembic
- **Machine Learning**: CatBoost, Scikit-Learn, Joblib, NumPy, Pandas
- **Natural Language Processing**: spaCy (`en_core_web_sm`), Sentence-Transformers (`all-MiniLM-L6-v2`), PyTorch
- **Frontend**: Streamlit, Plotly Express & Graph Objects, Requests
- **Database & Storage**: PostgreSQL 14+, Modular Disk / AWS S3 / Cloudflare R2 Upload Manager
- **DevOps & Containers**: Docker, Docker Compose, Linux multi-stage builds

---

## 📁 Project Directory Structure

```
safety-incident-prediction-nlp/
├── app.py                             # FastAPI application entrypoint & API endpoints
├── database.py                        # SQLAlchemy database engine and sessionmaker
├── models.py                          # SQLAlchemy ORM models & database views
├── file_storage.py                    # Secure document upload & orphan cleanup manager
├── import_historical_dataset.py       # Audited, chunked, idempotent CSV ingestion CLI
├── alembic.ini                        # Alembic database migration configuration
├── requirements.txt                   # Backend and ML pipeline dependencies
├── Dockerfile.backend                 # Production multi-stage Dockerfile for FastAPI
├── Dockerfile.frontend                # Production multi-stage Dockerfile for Streamlit
├── docker-compose.yml                 # Multi-service container orchestration manifest
├── render.yaml                        # PaaS infrastructure blueprint
├── .dockerignore                      # Docker build context exclusion rules
├── .gitignore                         # Git repository exclusion rules
├── .env.example                       # Environment configuration template
├── projectcontext.md                  # Comprehensive architectural and audit documentation
│
├── migrations/                        # Alembic schema migrations
│   ├── env.py                         # Alembic migration execution script
│   └── versions/
│       ├── 74771e86d71b_create_initial_database_schema.py
│       └── 8f853c4d65c8_add_historical_import_and_audit_schema.py
│
├── saved_models/                      # Serialized multi-target ML inference pipelines
│   ├── risk_level_pipeline.joblib           # CatBoost pipeline for Risk Level
│   ├── root_cause_category_pipeline.joblib  # CatBoost pipeline for Root Cause
│   └── severity_level_pipeline.joblib       # CatBoost pipeline for Severity Level
│
├── safety_incidents_cleaned.csv       # Verified historical training dataset (8,000 records)
├── safety_incidents_synthetic_raw.csv # Raw synthetic generation baseline dataset
├── SyntheticDtst.ipynb                # Data generation and CatBoost model training notebook
│
├── safety_incident_ui/                # Streamlit UI frontend
│   ├── app.py                         # Streamlit multi-page navigation router
│   ├── requirements.txt               # Frontend and NLP dependencies
│   ├── hybrid_nlp_prototype.py        # Semantic Transformer NLP extraction engine
│   ├── assets/styles.css              # Custom enterprise UI stylesheet
│   ├── components/                    # Reusable UI widgets and Plotly charts
│   ├── data/mock_data.py              # Offline demo and test dataset
│   ├── services/                      # API client and spaCy NER extraction services
│   ├── utils/                         # Validation and string formatting helpers
│   ├── views/                         # Application pages (Dashboard, Reporting, History, Analytics)
│   └── tests/                         # Pytest test suite for frontend & NLP
│
└── tests/                             # Root end-to-end integration test suite
    ├── test_backend_api.py            # API endpoint integration test suite
    ├── test_file_cleanup.py           # Upload storage retention test
    ├── test_system_verification.py    # Complete system workflow verification suite
    └── test_production_smoke.py       # Pre-deployment smoke test suite
```

---

## 🛠 Local Installation & Setup

### Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **PostgreSQL**: Version 14 or later (running locally or remotely)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/safety-incident-prediction-nlp.git
cd safety-incident-prediction-nlp
```

### 2. Configure Backend Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Frontend Virtual Environment (Recommended)
You can share the root environment or create an isolated environment for the UI:
```bash
cd safety_incident_ui
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# On Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..
```

---

## ⚙ Environment Variables

Copy the example configuration file:
```bash
cp .env.example .env
```

Edit `.env` to match your local or staging environment:
```ini
# PostgreSQL Database Connection Settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=safety_incident_db
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Application Settings
SAFETY_API_URL=http://127.0.0.1:8502
STORAGE_BACKEND=local
SAFETY_UPLOAD_DIR=private_uploads
SAFETY_DEMO_MODE=false
DISABLE_SQLALCHEMY_CEXT=1
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

---

## 🗄 Database Setup & Migrations

### 1. Initialize PostgreSQL Database
Create the application database:
```sql
CREATE DATABASE safety_incident_db;
```

### 2. Apply Database Migrations
Execute Alembic migrations to build the tables, views, and indexes:
```bash
alembic upgrade head
```

### 3. Ingest Historical Baseline Dataset
Run the safe, idempotent data ingestion script to import the 8,000 historical records:
```bash
# Validate CSV schema without modifying database
python import_historical_dataset.py --dry-run

# Ingest historical records into PostgreSQL
python import_historical_dataset.py --execute --yes
```

---

## 🚦 Running the Application

### Option A: Local Processes

**Terminal 1 — Start the FastAPI Backend**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8502 --reload
```
API Documentation will be available at `http://127.0.0.1:8502/docs`.

**Terminal 2 — Start the Streamlit Frontend**:
```bash
cd safety_incident_ui
streamlit run app.py --server.port 8501
```
The web dashboard will be available at `http://localhost:8501`.

---

### Option B: Docker Compose (Full Stack)

Ensure Docker is installed and running, then execute:
```bash
docker compose up -d --build
```
This builds and starts PostgreSQL, the FastAPI backend, and the Streamlit frontend with automated health checks and persistent volume mounts.

---

## 🧠 ML & NLP Pipeline Explanation

### 1. Multi-Target Machine Learning
The platform operates three independent CatBoost classifiers wrapped in Scikit-Learn pipelines:
- **Risk Level Model** (`saved_models/risk_level_pipeline.joblib`): Predicts `Low`, `Medium`, `High`, or `Critical`.
- **Severity Level Model** (`saved_models/severity_level_pipeline.joblib`): Predicts OSHA/internal severity tiers (e.g. `First Aid`, `Medical Treatment`, `Lost Time`, `Fatality`).
- **Root Cause Category Model** (`saved_models/root_cause_category_pipeline.joblib`): Categorizes causal factors (e.g. `Equipment Failure`, `Human Factor`, `Procedure Non-Compliance`, `Environmental Hazard`).

Features are composed of tabular categorical metadata (department, facility location, work shift) combined with NLP-extracted semantic entity indicators.

### 2. Hybrid NLP Entity Extraction
- **Rule-Based & Statistical NER**: Uses spaCy (`en_core_web_sm`) configured with custom entity rules to extract chemical names (e.g., Benzene, Chlorine, Hydrogen Sulfide), equipment identifiers (Pumps, Valves, Compressors, Flanges), and bodily injuries.
- **Semantic Transformer Embeddings**: Evaluates narrative similarity against known incident taxonomic vectors via `sentence-transformers/all-MiniLM-L6-v2` to classify root causes even when direct keyword matches are absent.

---

## 🧪 Testing & Quality Assurance

The project includes unit, regression, integration, and smoke test suites:

```bash
# Run UI & NLP regression test suite (100% non-mutating)
pytest safety_incident_ui/tests/test_core.py safety_incident_ui/tests/test_nlp_regression.py -v

# Run full frontend integration suite
pytest safety_incident_ui/tests/ -v

# Run backend API contract test suite
python test_backend_api.py

# Run production smoke test against running services
python test_production_smoke.py --api-url http://localhost:8502 --frontend-url http://localhost:8501
```

---

## 🚢 Deployment Options

- **Single VPS (Recommended)**: Ubuntu 24.04 with 2–4 GB RAM using `docker-compose.yml` and a Caddy reverse proxy for automatic SSL termination.
- **Managed Cloud PaaS**: Compatible with Render (`render.yaml`) or Railway combined with managed PostgreSQL (e.g., Neon or AWS RDS). *Minimum 1 GB RAM per container required to support ML and Transformer models.*

---

## 📸 Screenshots & UI Walkthrough

| Executive Dashboard | New Incident Report & Live NLP |
|:---:|:---:|
| ![Dashboard Placeholder](docs/images/dashboard_placeholder.png) | ![New Incident Placeholder](docs/images/reporting_placeholder.png) |
| *Real-time incident metrics, risk distribution, and departmental trends.* | *Raw narrative input with real-time entity extraction and validation.* |

| Root Cause Analysis & Prediction | Historical Dataset Explorer |
|:---:|:---:|
| ![Prediction Placeholder](docs/images/prediction_placeholder.png) | ![Historical Placeholder](docs/images/historical_placeholder.png) |
| *Multi-model prediction breakdown with confidence intervals.* | *Search and filter through 8,000+ audited historical incident records.* |

---

## 📄 License Recommendation

> [!NOTE]
> An open source license has not yet been applied. Recommended options for public distribution:
> - **Apache 2.0**: Recommended for enterprise and machine-learning projects requiring patent grant protections.
> - **MIT License**: Recommended for permissive, minimal-restriction public sharing.
>
> *Please consult project stakeholders to select the preferred open source license.*
