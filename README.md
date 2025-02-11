# README

## Guide to Running the Application

Before proceeding, create a Python environment and activate it:

```sh
python3 -m venv myenv
source myenv/bin/activate
```

### Prerequisites

#### 1. Install Dependencies

Run the following command to install all required dependencies:

```sh
pip install -r requirements.txt
```

#### 2. Set Up Google Cloud Credentials

To use Google Cloud services, follow these steps:

- Create a service account in **Google Cloud Platform (GCP)**.
- Enable **Speech-to-Text** and **Text-to-Speech** APIs.
- Download the **JSON key file**.
- Create a `.env` file in the project directory and add the following line, replacing `PATH_TO_JSON_FILE` with the actual path:

```sh
GOOGLE_APPLICATION_CREDENTIALS="PATH_TO_JSON_FILE"
```

The application will automatically load this key using `load_dotenv()`.

#### 3. Ensure Ollama is Installed and Running

- Install **Ollama** from the [official website](https://ollama.ai/).
- Download the required models by running the following commands:

```sh
ollama pull llama3.2:1b
ollama pull nomic-embed-text:latest
```

---

Your application is now ready to run!

