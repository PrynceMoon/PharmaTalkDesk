# Documentazione Tecnica: Dashboard di Consultazione Medicinali con Assistente Virtuale

## Indice
1. [Introduzione](#introduzione)
2. [Architettura del Sistema](#architettura-del-sistema)
3. [Componenti Principali](#componenti-principali)
4. [Stack Tecnologico](#stack-tecnologico)
5. [Funzionalità Implementate](#funzionalità-implementate)
6. [Integrazione del Modello LLM](#integrazione-del-modello-llm)
7. [Elaborazione del Linguaggio Naturale](#elaborazione-del-linguaggio-naturale)
8. [Gestione Multilingua](#gestione-multilingua)
9. [Ottimizzazione delle Prestazioni](#ottimizzazione-delle-prestazioni)
10. [Installazione e Configurazione](#installazione-e-configurazione)
11. [Guida Utente](#guida-utente)
12. [Limitazioni e Sviluppi Futuri](#limitazioni-e-sviluppi-futuri)
13. [Appendice: Struttura del Codice](#appendice-struttura-del-codice)

## Introduzione

La Dashboard di Consultazione Medicinali è un'applicazione interattiva progettata per fornire agli utenti un accesso semplificato alle informazioni sui medicinali attraverso un'interfaccia grafica intuitiva e un assistente virtuale basato su LLM (Large Language Model). Il sistema integra una dashboard di visualizzazione dati con un chatbot intelligente che sfrutta il modello Mistral scaricato tramite Ollama per rispondere a domande in linguaggio naturale sui medicinali.

Il sistema è stato sviluppato con l'obiettivo di:
- Fornire informazioni dettagliate sui medicinali in modo accessibile
- Consentire la consultazione dei dati tramite filtri e ricerche
- Offrire supporto multilingua (italiano e inglese)
- Integrare un assistente intelligente per consulenza sui farmaci
- Visualizzare statistiche e analisi sui medicinali disponibili

Questa documentazione fornisce una panoramica completa dell'architettura, delle funzionalità e delle tecnologie utilizzate nell'implementazione del sistema.

## Architettura del Sistema

L'architettura del sistema è composta da diverse componenti integrate che lavorano insieme per fornire un'esperienza utente completa:

```
┌──────────────────────────────────────────┐
│            Interfaccia Streamlit         │
├─────────────┬────────────────┬───────────┤
│ Dashboard   │ Chatbot UI     │ Dettagli  │
│ Medicinali  │                │ Medicinale│
└─────────────┴────────────────┴───────────┘
         │              │
         │              │
         ▼              ▼
┌───────────────┐ ┌─────────────────────┐
│ Dataset       │ │ Server FastAPI       │
│ Medicinali    │ │ (Backend)           │
└───────────────┘ └─────────────────────┘
         │                 │
         │                 │
         ▼                 ▼
┌───────────────┐ ┌─────────────────────┐
│ Deep          │ │ Modello LLM         │
│ Translator    │ │ (Mistral via Ollama)│
└───────────────┘ └─────────────────────┘
```

Il sistema adotta un'architettura client-server dove:
- Streamlit fornisce l'interfaccia utente e la visualizzazione
- FastAPI gestisce le richieste API del chatbot
- Ollama fornisce l'accesso al modello Mistral per l'elaborazione del linguaggio naturale
- Deep Translator si occupa della traduzione automatica dei contenuti

## Componenti Principali

### 1. Dashboard di Visualizzazione
La dashboard presenta i dati sui medicinali in formato tabulare con funzionalità di filtro e ricerca. Visualizza statistiche aggregate sui medicinali tramite grafici interattivi.

### 2. Assistente Virtuale
Un chatbot basato sul modello Mistral che risponde a domande sui medicinali in linguaggio naturale. L'assistente analizza semanticamente le domande dell'utente e recupera informazioni pertinenti dal dataset.

### 3. Sistema di Traduzione
Componente che gestisce la traduzione automatica dei dati sui medicinali dall'inglese all'italiano mediante l'API Google Translator.

### 4. Backend FastAPI
Server API che gestisce le richieste dell'assistente virtuale, elabora le query dell'utente e comunica con il modello LLM per generare risposte appropriate.

### 5. Visualizzatore Dettagli Medicinali
Sezione dedicata alla visualizzazione approfondita dei dettagli di un singolo medicinale selezionato dall'utente.

## Stack Tecnologico

Il sistema è stato sviluppato utilizzando le seguenti tecnologie:

### Frontend
- **Streamlit**: Framework Python per la creazione di applicazioni web interattive
- **Plotly Express**: Libreria per la creazione di grafici interattivi
- **Pandas**: Libreria per la manipolazione e l'analisi dei dati

### Backend
- **FastAPI**: Framework per la creazione di API RESTful ad alte prestazioni
- **Uvicorn**: Server ASGI ad alte prestazioni
- **Pydantic**: Libreria per la validazione dei dati e le impostazioni

### Elaborazione del Linguaggio Naturale
- **Ollama**: Strumento per il deployment locale di modelli linguistici
- **Mistral**: Modello linguistico per la generazione di testo e l'assistenza conversazionale

### Traduzione e Utilità
- **Deep Translator**: Libreria per la traduzione automatica dei testi
- **Threading**: Modulo per la gestione di operazioni concorrenti
- **AsyncIO e Nest AsyncIO**: Librerie per la programmazione asincrona

## Funzionalità Implementate

### Gestione dei Dati
- Caricamento e preprocessing del dataset medicinali
- Traduzione automatica dei dati dall'inglese all'italiano
- Caching dei dati per ottimizzare le prestazioni
- Filtrazione dinamica dei dati in base ai criteri di ricerca

### Visualizzazione Dati
- Tabella interattiva con i dettagli dei medicinali
- Grafici a torta per la visualizzazione delle statistiche di recensione
- Sezione dettagli per l'approfondimento sul singolo medicinale
- Indicatori metrici per i principali parametri del medicinale selezionato

### Interfaccia Chatbot
- Sistema di chat interattivo con cronologia dei messaggi
- Analisi semantica delle domande dell'utente
- Ricerca intelligente dei medicinali rilevanti
- Generazione di risposte contestuali basate sui dati disponibili
- Cache delle risposte per migliorare i tempi di risposta

### Sistema Multilingua
- Toggle per la selezione della lingua (italiano/inglese)
- Visualizzazione dinamica dei contenuti nella lingua selezionata
- Traduzione automatica di utilizzi ed effetti collaterali

### Gestione Server
- Avvio automatico del server FastAPI in background
- Rilevamento dinamico delle porte disponibili
- Gestione delle risorse asincrone con AsyncIO
- Meccanismi di sicurezza per la gestione degli errori

## Integrazione del Modello LLM

Il sistema integra il modello linguistico Mistral tramite Ollama, un framework per l'esecuzione locale di LLM. L'integrazione avviene attraverso i seguenti passaggi:

### Configurazione del Modello
```python
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
```

### Sistema di Prompt Engineering
Il sistema utilizza un sistema di prompt strutturato per ottimizzare le risposte del modello:

```python
DEFAULT_SYSTEM_PROMPT = """Sei un assistente medico esperto che risponde SEMPRE in italiano. 
Il tuo compito è aiutare gli utenti fornendo informazioni accurate sui medicinali basandoti sui dati forniti. 
Rispondi in modo chiaro e professionale, utilizzando ESCLUSIVAMENTE la lingua italiana."""
```

### Generazione delle Risposte
La funzione `generate_ollama_response` si occupa di inviare richieste al modello e processare le risposte:

```python
def generate_ollama_response(prompt, system_prompt, max_tokens=500):
    # Configurazione della richiesta
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7
        }
    }
    
    # Invio della richiesta all'API
    response = requests.post(OLLAMA_ENDPOINT, headers=headers, json=data)
    # Elaborazione della risposta
    # ...
```

### Verifica Disponibilità del Modello
Il sistema verifica che Ollama sia in esecuzione prima di tentare di generare risposte:

```python
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False
```

## Elaborazione del Linguaggio Naturale

### Analisi Semantica delle Domande
Il sistema implementa un'analisi semantica delle domande dell'utente per identificare il tipo di richiesta:

```python
query_terms = set(prompt.lower().split())
query_type = {
    'sintomi': any(term in query_terms for term in ['sintomo', 'sintomi', 'male', 'dolore', 'disturbo']),
    'effetti_collaterali': any(term in query_terms for term in ['effetto', 'collaterale', 'reazione']),
    'dosaggio': any(term in query_terms for term in ['dose', 'dosaggio', 'quanto', 'assumere', 'assunzione']),
    'controindicazioni': any(term in query_terms for term in ['controindicazione', 'controindicato', 'evitare']),
    'generale': any(term in query_terms for term in ['cosa', 'come', 'quando', 'perché'])
}
```

### Ricerca Avanzata nei Medicinali
Il sistema implementa un algoritmo di ricerca che assegna punteggi di rilevanza ai medicinali in base alla corrispondenza con i termini della query:

```python
# Calcolo diversi tipi di rilevanza
name_relevance = sum(1 for term in query_terms if term in med_name) * 2
uses_relevance = sum(1 for term in query_terms if term in med_uses)
effects_relevance = sum(1 for term in query_terms if term in med_effects)
composition_relevance = sum(1 for term in query_terms if term in med_composition)

total_relevance = name_relevance + uses_relevance + effects_relevance + composition_relevance
```

### Sistema di Cache delle Risposte
Per ottimizzare le prestazioni, il sistema implementa un meccanismo di cache delle risposte basato sulla similarità semantica delle domande:

```python
def calculate_similarity(query1, query2):
    """Calcola la similarità tra due query"""
    query1_terms = set(query1.lower().split())
    query2_terms = set(query2.lower().split())
    intersection = query1_terms.intersection(query2_terms)
    union = query1_terms.union(query2_terms)
    return len(intersection) / len(union) if union else 0

def get_cached_response(prompt, threshold=0.8):
    """Cerca una risposta simile nella cache"""
    for cached_prompt, response in st.session_state.response_cache.items():
        if calculate_similarity(prompt, cached_prompt) > threshold:
            return response
    return None
```

## Gestione Multilingua

### Toggle della Lingua
L'applicazione implementa un sistema di selezione della lingua tramite un toggle nella sidebar:

```python
is_italian = st.sidebar.toggle("🌍 EN / IT", value=True, key="language_toggle")
st.session_state.language = "Italiano" if is_italian else "English"
```

### Traduzione Automatica del Dataset
Il sistema traduce automaticamente i campi principali del dataset dall'inglese all'italiano:

```python
@st.cache_data
def load_data():
    # Verifica se esiste già un file con le traduzioni
    if os.path.exists('Medicine_Details_Translated.csv'):
        df = pd.read_csv('Medicine_Details_Translated.csv')
        # Verifica che tutte le colonne necessarie siano presenti
        required_columns = ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 
                          'Image URL', 'Manufacturer', 'Excellent Review %', 
                          'Average Review %', 'Poor Review %', 'Uses_IT', 'Side_effects_IT']
        if all(col in df.columns for col in required_columns):
            return df
    
    # Carica il dataset originale e procede con la traduzione
    # ...
```

### Visualizzazione Contenuti Multilingua
L'interfaccia si adatta dinamicamente alla lingua selezionata:

```python
display_df = filtered_df[['Medicine Name', 'Composition', 'Uses_IT', 'Side_effects_IT']].copy()
display_df.columns = ['Nome Medicinale', 'Composizione', 'Utilizzi', 'Effetti Collaterali']
if st.session_state.language == "English":
    display_df = filtered_df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].copy()
    display_df.columns = ['Medicine Name', 'Composition', 'Uses', 'Side Effects']
```

## Ottimizzazione delle Prestazioni

### Caching dei Dati
Il sistema utilizza la funzionalità di caching di Streamlit per evitare il caricamento ripetuto dei dati:

```python
@st.cache_data
def load_data():
    # ...
```

### Threading e Operazioni Asincrone
L'applicazione implementa un sistema di threading per gestire operazioni parallele:

```python
# Avvio del server FastAPI in un thread separato
if 'fastapi_thread' not in st.session_state:
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    st.session_state.fastapi_thread = fastapi_thread
    time.sleep(1)  # Breve pausa per permettere l'avvio del server
```

### Rilevamento Porte Disponibili
L'applicazione implementa un meccanismo per la ricerca dinamica di porte disponibili:

```python
def find_free_port(start_port=8000, max_port=8100):
    """Trova una porta libera partendo da start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None
```

### Rate Limiting per le Traduzioni
Il sistema implementa un meccanismo di rate limiting per evitare blocchi durante la traduzione:

```python
def translate_text(text, progress_bar, status_text, current_item, total_items, current_column):
    # ...
    time.sleep(0.5)  # Delay per evitare blocchi
    # ...
```

## Installazione e Configurazione

### Prerequisiti
- Python 3.8+
- Ollama installato e configurato con il modello Mistral
- Connessione Internet per la traduzione

### Dipendenze Python
```bash
pip install streamlit pandas plotly deep-translator fastapi uvicorn nest-asyncio pydantic requests
```

### Configurazione di Ollama
1. Installare Ollama seguendo le istruzioni sul sito ufficiale
2. Scaricare il modello Mistral:
```bash
ollama pull mistral
```

### Preparazione dei Dati
Il sistema richiede un file CSV denominato `Medicine_Details.csv` contenente i seguenti campi:
- Medicine Name
- Composition
- Uses
- Side_effects
- Image URL
- Manufacturer
- Excellent Review %
- Average Review %
- Poor Review %

### Avvio dell'Applicazione
```bash
streamlit run chatbot_medico.py
```

## Guida Utente

### Navigazione nell'Interfaccia
L'interfaccia utente è suddivisa in diverse sezioni:

1. **Barra Laterale**: Contiene il toggle della lingua e i filtri di ricerca
2. **Lista Medicinali**: Tabella con i principali dati sui medicinali disponibili
3. **Statistiche**: Grafici che visualizzano le statistiche sui medicinali
4. **Assistente Virtuale**: Chatbot interattivo per domande sui medicinali
5. **Dettagli Medicinale**: Visualizzazione dettagliata del medicinale selezionato

### Ricerca dei Medicinali
Per cercare un medicinale specifico:
1. Utilizzare il campo di ricerca nella barra laterale
2. Inserire il nome o parte del nome del medicinale
3. La tabella si aggiornerà automaticamente mostrando i risultati corrispondenti

### Utilizzo dell'Assistente Virtuale
Per interagire con l'assistente:
1. Digitare una domanda nella casella di input in fondo alla sezione chatbot
2. Premere Invio o cliccare sull'icona per inviare la domanda
3. L'assistente analizzerà la domanda e fornirà una risposta basata sui dati disponibili

### Visualizzazione dei Dettagli
Per visualizzare i dettagli di un medicinale:
1. Selezionare il medicinale desiderato dal menu a tendina nella sezione "Dettagli Medicinale"
2. I dettagli completi del medicinale selezionato verranno visualizzati automaticamente

### Cambio della Lingua
Per cambiare la lingua dell'interfaccia:
1. Utilizzare il toggle "🌍 EN / IT" nella barra laterale
2. Tutti i contenuti dell'interfaccia verranno aggiornati nella lingua selezionata

## Limitazioni e Sviluppi Futuri

### Limitazioni Attuali
- **Dipendenza da Ollama**: Il sistema richiede che Ollama sia installato e configurato localmente
- **Limitazioni del Modello**: Il modello Mistral, sebbene potente, ha limitazioni nella comprensione medica specialistica
- **Performance su Dataset Grandi**: La traduzione e l'elaborazione di dataset molto grandi può risultare lenta
- **Supporto Lingue**: Attualmente supporta solo italiano e inglese

### Sviluppi Futuri
- **Integrazione Database**: Sostituire il sistema basato su CSV con un database relazionale
- **Supporto Multimodale**: Aggiungere il supporto per l'analisi di immagini dei medicinali
- **Espansione LLM**: Supporto per modelli linguistici alternativi e più specializzati
- **API Esterne**: Integrazione con fonti di dati esterne per informazioni mediche più complete
- **Selezione Modelli**: Implementare un sistema per selezionare diversi modelli LLM in base alle esigenze
- **Sistema di Feedback**: Aggiungere un meccanismo per raccogliere feedback dagli utenti sulle risposte

## Appendice: Struttura del Codice

Il codice sorgente è organizzato nelle seguenti sezioni principali:

### Importazioni e Configurazione Iniziale
Importazione delle librerie necessarie e configurazione dell'ambiente Streamlit.

### Configurazione dell'Event Loop per FastAPI
Implementazione del sistema asincrono per la gestione del server FastAPI.

### Definizioni delle Funzioni di Utilità
Funzioni per la ricerca di porte libere, traduzione, verifica dello stato del server, ecc.

### Configurazione FastAPI
Definizione dell'applicazione FastAPI e degli endpoint per il chatbot.

### Funzioni di Traduzione e Caricamento Dati
Implementazione del sistema di traduzione e caricamento del dataset.

### Funzioni per l'Elaborazione del Linguaggio Naturale
Algoritmi per l'analisi semantica delle domande e la generazione delle risposte.

### Definizione dell'Interfaccia Utente
Implementazione delle diverse sezioni dell'interfaccia Streamlit.

### Sistema di Cache delle Risposte
Implementazione del meccanismo di cache per ottimizzare le prestazioni del chatbot.

---

*Questa documentazione è stata creata per un progetto universitario e rappresenta una panoramica completa del sistema di Dashboard di Consultazione Medicinali con Assistente Virtuale basato su LLM. Per ulteriori informazioni o supporto, contattare il team di sviluppo.*
