# 🤖 POC Chatbot RAG - Use Case Proditec

Ce dépôt contient une preuve de concept (POC) d'un **Assistant Documentaire Intelligent** basé sur l'architecture **RAG (Retrieval-Augmented Generation)**. 

Conçu dans le cadre du workshop **AI4Industry** pour le cas d'usage **Proditec**, cet outil permet d'interroger en langage naturel une base de connaissances constituée de documents techniques, manuels et rapports internes.

## 🚀 Fonctionnalités Clés

- **Interface Conversationnelle (Gradio)** : Chatbot intuitif pour poser des questions.
- **Support Multi-Formats** : Ingestion de fichiers PDF, DOCX, XLSX, PPIX, etc. Côté backend
- **RAG Local & Sécurisé** :
  - **Ollama** : Utilisation de LLM Open-Source Mistral en local.
  - **ChromaDB** : Base de données vectorielle persistante via Docker.
- **Transparence** : Citations précises des sources et affichage des extraits utilisés pour générer chaque réponse.
- **Outils de Traitement de Données** : Scripts avancés pour la conversion en masse et le nettoyage de documents (OCR, fusion de lignes brisées, suppression du bruit).

---

## 🏗️ Architecture Technique

Le projet repose sur la stack technique suivante :

- **Frontend** : [Gradio](https://www.gradio.app/) (Interface Web).
- **Orchestration RAG** : [LangChain](https://www.langchain.com/).
- **LLM Engine** : [Ollama](https://ollama.com/) (pour l'inférence locale).
- **Vector Store** : [ChromaDB](https://www.trychroma.com/) (stockage des embeddings).
- **Conversion** : `MarkItDown` (Microsoft) pour la conversion universelle de documents vers Markdown.

---
## Modèles d'IA utilisés
Modèle LLM : mistral-7b
Modèle Embedding : intfloat/e5-mistral-7b-instruct

## 🛠️ Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

1.  **Python 3.12**
2.  **Ollama** (installé et fonctionnel sur votre machine)

> **Note** : Assurez-vous d'avoir téléchargé le modèle dans Ollama au préalable : 

```bash
ollama pull mistral
```

---

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/MatysChampeyrol/use_case_proditec

cd use_case_proditec
```

### 2. Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

Il existe deux façons d'utiliser le projet : via l'interface web (recommandé) ou via les scripts de traitement par lot.

### Option A : Interface Web (Chatbot)

Lancez l'application principale :

```bash
python -m src.main.run
```

L'interface sera accessible à l'adresse **http://localhost:7860**.

**Étapes :**
1.  **Charger un document** : Dans le panneau de gauche, déposez un fichier (PDF, DOCX, etc.).
2.  **Configurer** (optionnel) : Ajustez la taille des "chunks" (morceaux de texte) et l'overlap.
3.  **Indexer** : Cliquez sur **🚀 Indexer**. Le document est traité et ajouté à ChromaDB.
4.  **Discuter** : Posez vos questions dans le chat à droite. L'IA vous répondra en citant ses sources.

---

## 📂 Structure du Projet

```plaintext
use_case_proditec/
├── docker-compose.yml       # Configuration Docker pour ChromaDB
├── requirements.txt         # Dépendances Python
├── convert_docs.py          # Script de conversion de documents (Batch)
├── markdown_parser.py       # Script de nettoyage Markdown (Batch)
├── src/
│   └── main/
│       ├── run.py           # Point d'entrée de l'application (Gradio)
│       └── service/
│           └── rag_service.py # Logique métier RAG (ingestion, requêtage)
├── uploaded_docs/           # Dossier temporaire pour les uploads
└── README.md                # Documentation du projet
```
