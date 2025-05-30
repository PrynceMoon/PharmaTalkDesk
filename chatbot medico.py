# ============= IMPORTAZIONI =============
# Librerie di base per l'interfaccia web e la manipolazione dei dati
import streamlit as st  # Framework per creare app web
import sys  # Funzionalità di sistema
import pandas as pd  # Manipolazione e analisi dei dati
import plotly.express as px  # Creazione di grafici interattivi
from deep_translator import GoogleTranslator  # Traduzione automatica del testo
import time  # Gestione del tempo e dei ritardi
import os  # Operazioni sul sistema operativo
import json  # Gestione di dati JSON
import requests  # Richieste HTTP
from fastapi import FastAPI, BackgroundTasks  # Framework per API
import uvicorn  # Server ASGI per FastAPI
import threading  # Gestione dei thread
import asyncio  # Programmazione asincrona
import nest_asyncio  # Gestione di loop asincroni annidati
from pydantic import BaseModel  # Validazione dei dati
import gc  # Garbage collector
import socket  # Operazioni di rete
import logging  # Sistema di logging

# ============= CONFIGURAZIONE LOGGING =============
# Riduce il livello di logging per evitare messaggi di debug non necessari
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ============= CONFIGURAZIONE STREAMLIT =============
# Impostazione della configurazione della pagina web
st.set_page_config(
    page_title="Dashboard Medicinali",  # Titolo della pagina
    page_icon="💊",  # Icona della pagina
    layout="wide"  # Layout a schermo intero
)

# Inizializzazione dello stato di caricamento
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Inizializzazione dello stato di elaborazione del chatbot
if 'chatbot_processing' not in st.session_state:
    st.session_state.chatbot_processing = False

# ============= CONFIGURAZIONE ASINCRONA =============
# Configurazione semplificata per evitare warning asincroni
nest_asyncio.apply()  # Applica il supporto per loop annidati

# Funzione per trovare una porta libera per il server
def find_free_port(start_port=8000, max_port=8100):
    """Cerca una porta disponibile nell'intervallo specificato"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

# ============= CONFIGURAZIONE FASTAPI =============
# Inizializzazione dell'app FastAPI
app = FastAPI()

# Variabile globale per la porta del server
fastapi_port = None

# Funzione per avviare il server FastAPI
def run_fastapi():
    global fastapi_port
    try:
        # Trova una porta libera per il server
        free_port = find_free_port()
        if not free_port:
            st.warning("Non è stato possibile trovare una porta libera per il server FastAPI. L'applicazione continuerà a funzionare ma alcune funzionalità potrebbero essere limitate.")
            return
        
        fastapi_port = free_port
        
        # Configura e avvia il server in modo sincrono
        config = uvicorn.Config(app, host="127.0.0.1", port=fastapi_port, log_level="error")
        server = uvicorn.Server(config)
        
        try:
            # Crea un nuovo event loop per questo thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())
        except SystemExit:
            pass  # Gestisce l'uscita normale del server
        except Exception as e:
            # Ignora gli errori di chiusura del server
            pass
        finally:
            # Chiudi il loop correttamente
            try:
                loop.close()
            except:
                pass
    except Exception as e:
        # Riduci il livello di logging per evitare spam di errori
        pass

# Avvio del server FastAPI in un thread separato con gestione migliorata
if 'fastapi_started' not in st.session_state:
    st.session_state.fastapi_started = False

if not st.session_state.fastapi_started:
    try:
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_started = True
        time.sleep(0.5)  # Breve pausa per permettere l'avvio del server
    except Exception:
        # Ignora errori di avvio del server FastAPI
        pass

# ============= STILE CSS =============
# Definizione dello stile per il pulsante di cambio lingua e elementi disabilitati
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #f0f2f6;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Stile per elementi disabilitati durante l'elaborazione */
    .disabled-element {
        pointer-events: none;
        opacity: 0.6;
    }
    
    /* Stile per l'indicatore di caricamento */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.7);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# ============= GESTIONE DELLA LINGUA =============
# Inizializzazione della lingua predefinita
if 'language' not in st.session_state:
    st.session_state.language = "Italiano"

# Contenitore per il pulsante della lingua con gestione dello stato di disabilitazione
if not st.session_state.chatbot_processing:
    is_italian = st.sidebar.toggle("🌍 EN / IT", value=True, key="language_toggle")
    st.session_state.language = "Italiano" if is_italian else "English"
else:
    # Versione disabilitata del toggle durante l'elaborazione
    st.sidebar.markdown('<div class="disabled-element">', unsafe_allow_html=True)
    st.sidebar.toggle("🌍 EN / IT", value=True if st.session_state.language == "Italiano" else False, disabled=True, key="language_toggle_disabled")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ============= GESTIONE DEL TESTO MULTILINGUA =============
def get_text(key):
    """Restituisce il testo nella lingua selezionata"""
    # Dizionario delle traduzioni per tutte le stringhe dell'interfaccia
    translations = {
        "page_title": {
            "it": "Dashboard Consultazione Medicinali",
            "en": "Medicines Consultation Dashboard"
        },
        "search": {
            "it": "Ricerca",
            "en": "Search"
        },
        "search_medicine": {
            "it": "Cerca medicinale per nome:",
            "en": "Search medicine by name:"
        },
        "medicine_list": {
            "it": "Lista Medicinali",
            "en": "Medicines List"
        },
        "medicine_name": {
            "it": "Nome Medicinale",
            "en": "Medicine Name"
        },
        "composition": {
            "it": "Composizione",
            "en": "Composition"
        },
        "uses": {
            "it": "Utilizzi",
            "en": "Uses"
        },
        "side_effects": {
            "it": "Effetti Collaterali",
            "en": "Side Effects"
        },
        "statistics": {
            "it": "Statistiche",
            "en": "Statistics"
        },
        "reviews_distribution": {
            "it": "Distribuzione Media delle Recensioni",
            "en": "Average Reviews Distribution"
        },
        "excellent": {
            "it": "Eccellenti",
            "en": "Excellent"
        },
        "average": {
            "it": "Nella Media",
            "en": "Average"
        },
        "poor": {
            "it": "Scarse",
            "en": "Poor"
        },
        "virtual_assistant": {
            "it": "💬 Assistente Virtuale (Mistral)",
            "en": "💬 Virtual Assistant (Mistral)"
        },
        "ask_question": {
            "it": "Fai una domanda sui medicinali...",
            "en": "Ask a question about medicines..."
        },
        "medicine_details": {
            "it": "Dettagli Medicinale",
            "en": "Medicine Details"
        },
        "select_medicine": {
            "it": "Seleziona un medicinale per vedere i dettagli:",
            "en": "Select a medicine to see details:"
        },
        "name": {
            "it": "Nome",
            "en": "Name"
        },
        "manufacturer": {
            "it": "Produttore",
            "en": "Manufacturer"
        },
        "excellent_reviews": {
            "it": "Recensioni Eccellenti",
            "en": "Excellent Reviews"
        },
        "average_reviews": {
            "it": "Recensioni Nella Media",
            "en": "Average Reviews"
        },
        "poor_reviews": {
            "it": "Recensioni Scarse",
            "en": "Poor Reviews"
        },
        "overview": {
            "it": "📊 Panoramica Generale",
            "en": "📊 General Overview"
        },
        "total_medicines": {
            "it": "Totale Medicinali",
            "en": "Total Medicines"
        },
        "unique_manufacturers": {
            "it": "Produttori Unici",
            "en": "Unique Manufacturers"
        },
        "avg_excellent_reviews": {
            "it": "Media Eccellenti",
            "en": "Avg Excellent Reviews"
        },
        "best_medicine": {
            "it": "Miglior Medicinale",
            "en": "Best Medicine"
        },
    }
    
    # Determina la lingua corrente e restituisce la traduzione appropriata
    lang = "it" if st.session_state.language == "Italiano" else "en"
    return translations.get(key, {}).get(lang, key)

# ============= TITOLO PRINCIPALE =============
st.title("💊 " + get_text("page_title"))
st.markdown("---")

# ============= CONFIGURAZIONE SIDEBAR E FILTRI =============
# Aggiunta dell'header dei filtri nella sidebar
st.sidebar.header(get_text("search"))

# Inizializzazione della variabile search_term
search_term = ""

# Campo di ricerca per nome del medicinale con gestione dello stato di disabilitazione
if not st.session_state.chatbot_processing:
    search_term = st.sidebar.text_input(get_text("search_medicine"), key="search_input")
else:
    st.sidebar.markdown('<div class="disabled-element">', unsafe_allow_html=True)
    search_term = st.sidebar.text_input(get_text("search_medicine"), value="", disabled=True, key="search_input_disabled")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ============= CONFIGURAZIONE OLLAMA =============
# Impostazione dell'endpoint e del modello per il chatbot
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# ============= DEFINIZIONE MODELLO DATI =============
# Classe per la struttura delle richieste API del chatbot
class ChatRequest(BaseModel):
    prompt: str
    context: str

# ============= CONFIGURAZIONE PROMPT DI SISTEMA =============
# Prompt predefinito per il comportamento del chatbot
DEFAULT_SYSTEM_PROMPT = """Sei un assistente medico esperto che risponde SEMPRE in italiano. 
Il tuo compito è aiutare gli utenti fornendo informazioni accurate sui medicinali basandoti sui dati forniti. 
Rispondi in modo chiaro e professionale, utilizzando ESCLUSIVAMENTE la lingua italiana."""

# ============= ENDPOINT FASTAPI =============
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Gestisce le richieste di chat in arrivo"""
    try:
        # Genera la risposta utilizzando Ollama
        response = generate_ollama_response(request.prompt, DEFAULT_SYSTEM_PROMPT)
        return {"response": response}
    except Exception as e:
        return {"response": f"Errore nella generazione della risposta: {str(e)}"}

# ============= FUNZIONI DI UTILITÀ =============
def check_ollama_status():
    """Verifica se il servizio Ollama è attivo e raggiungibile"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

# ============= GESTIONE CACHE CHATBOT =============
# Inizializzazione della cache per le risposte del chatbot
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

# Limite massimo per la cache per evitare accumulo eccessivo di memoria
MAX_CACHE_SIZE = 50

def calculate_similarity(query1, query2):
    """Calcola la similarità tra due query basandosi sui termini comuni"""
    query1_terms = set(query1.lower().split())
    query2_terms = set(query2.lower().split())
    intersection = query1_terms.intersection(query2_terms)
    union = query1_terms.union(query2_terms)
    return len(intersection) / len(union) if union else 0

def get_cached_response(prompt, threshold=0.8):
    """Cerca una risposta simile nella cache basandosi sulla similarità delle query"""
    for cached_prompt, response in st.session_state.response_cache.items():
        if calculate_similarity(prompt, cached_prompt) > threshold:
            return response
    return None

def add_to_cache(prompt, response):
    """Aggiunge una risposta alla cache con gestione del limite massimo"""
    # Se la cache è piena, rimuovi l'elemento più vecchio
    if len(st.session_state.response_cache) >= MAX_CACHE_SIZE:
        # Rimuovi il primo elemento (il più vecchio)
        first_key = next(iter(st.session_state.response_cache))
        del st.session_state.response_cache[first_key]
        # Forza garbage collection
        gc.collect()
    
    # Aggiungi la nuova risposta
    st.session_state.response_cache[prompt] = response

# Funzione per tradurre il testo con gestione degli errori e rate limiting
def translate_text(text, progress_bar, status_text, current_item, total_items, current_column):
    if pd.isna(text):
        return text
    try:
        # Aggiorna lo stato di avanzamento
        progress = (current_item + 1) / total_items
        if current_column == 'Uses':
            progress = progress * 0.5  # Prima metà della barra
        else:
            progress = 0.5 + (progress * 0.5)  # Seconda metà della barra
        
        progress_bar.progress(progress)
        status_text.text(f"Traduzione in corso: {current_column} ({current_item + 1}/{total_items})")
        
        time.sleep(0.5)  # Delay per evitare blocchi
        return GoogleTranslator(source='en', target='it').translate(str(text))
    except Exception as e:
        st.warning(f"Errore nella traduzione dell'elemento {current_item + 1}: {str(e)}")
        return text

# Caricamento dei dati
@st.cache_data(show_spinner=False, ttl=3600)  # Cache con TTL di 1 ora per evitare accumulo
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
        else:
            # Se mancano colonne, ricarica e ritraduce
            if os.path.exists('Medicine_Details_Translated.csv'):
                os.remove('Medicine_Details_Translated.csv')
    
    # Carica il dataset originale
    df = pd.read_csv('Medicine_Details.csv')
    
    # Crea una barra di progresso per la traduzione
    progress_container = st.container()
    with progress_container:
        st.markdown("### Traduzione del Dataset in corso...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        items_status = st.empty()
        
        total_rows = len(df)
        
        try:
            # Traduzione degli utilizzi
            uses_translations = []
            for idx, text in enumerate(df['Uses']):
                translated = translate_text(
                    text, 
                    progress_bar, 
                    status_text, 
                    idx, 
                    total_rows,
                    'Uses'
                )
                uses_translations.append(translated)
                items_status.text(f"✓ Utilizzi tradotti: {idx + 1}/{total_rows}")
                
                # Garbage collection ogni 50 elementi per liberare memoria
                if idx % 50 == 0:
                    gc.collect()
                    
            df['Uses_IT'] = uses_translations
            
            # Traduzione degli effetti collaterali
            side_effects_translations = []
            for idx, text in enumerate(df['Side_effects']):
                translated = translate_text(
                    text, 
                    progress_bar, 
                    status_text, 
                    idx, 
                    total_rows,
                    'Side_effects'
                )
                side_effects_translations.append(translated)
                items_status.text(f"✓ Utilizzi tradotti: {total_rows}/{total_rows}\n✓ Effetti collaterali tradotti: {idx + 1}/{total_rows}")
                
                # Garbage collection ogni 50 elementi per liberare memoria
                if idx % 50 == 0:
                    gc.collect()
                    
            df['Side_effects_IT'] = side_effects_translations
            
            # Verifica che tutte le colonne siano presenti prima di salvare
            columns_check = ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 
                           'Image URL', 'Manufacturer', 'Excellent Review %', 
                           'Average Review %', 'Poor Review %', 'Uses_IT', 'Side_effects_IT']
            
            for col in columns_check:
                if col not in df.columns:
                    raise Exception(f"Colonna mancante: {col}")
            
            # Salva il DataFrame tradotto con tutte le colonne
            df.to_csv('Medicine_Details_Translated.csv', index=False)
            
            # Mostra un riepilogo delle colonne
            st.success("✅ Traduzione completata! Il dataset contiene le seguenti colonne:")
            st.write("Colonne originali:")
            st.write("- Medicine Name (Nome Medicinale)")
            st.write("- Composition (Composizione)")
            st.write("- Uses (Utilizzi in inglese)")
            st.write("- Side_effects (Effetti collaterali in inglese)")
            st.write("- Image URL (URL dell'immagine)")
            st.write("- Manufacturer (Produttore)")
            st.write("- Excellent/Average/Poor Review % (Percentuali recensioni)")
            st.write("\nColonne tradotte aggiunte:")
            st.write("- Uses_IT (Utilizzi in italiano)")
            st.write("- Side_effects_IT (Effetti collaterali in italiano)")
            
            # Pulizia UI
            progress_container.empty()
            
            # Garbage collection finale
            gc.collect()
            
        except Exception as e:
            st.error(f"Errore durante la traduzione: {str(e)}")
            # In caso di errore, usa il DataFrame originale
            df['Uses_IT'] = df['Uses']
            df['Side_effects_IT'] = df['Side_effects']
    
    return df

# Applicazione dei filtri
filtered_df = load_data()
if search_term:
    filtered_df = filtered_df[filtered_df['Medicine Name'].str.contains(search_term, case=False, na=False)]

# ============= GENERAZIONE RISPOSTE OLLAMA =============
def generate_ollama_response(prompt, system_prompt, max_tokens=500):
    """Genera una risposta utilizzando il modello Ollama"""
    try:
        # Configurazione degli headers per la richiesta
        headers = {
            "Content-Type": "application/json"
        }
        
        # Preparazione dei dati per la richiesta
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,  # Disattiva lo streaming per evitare output parziali
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7  # Controlla la creatività delle risposte
            }
        }
        
        # Verifica se esiste una risposta in cache
        cached_response = get_cached_response(prompt)
        if cached_response:
            return cached_response
        
        # Se non c'è cache, genera una nuova risposta
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()['response']
            # Memorizza la risposta nella cache
            add_to_cache(prompt, result)
            return result
        else:
            return f"Errore nella chiamata a Ollama: {response.status_code}"
    except Exception as e:
        return f"Errore nella generazione della risposta: {str(e)}"

# ============= GESTIONE RISPOSTA AI =============
def get_local_ai_response(prompt, medicine_data):
    """Genera una risposta AI basata sui dati dei medicinali disponibili"""
    try:
        # Verifica lo stato di Ollama
        if not check_ollama_status():
            return "Errore: Ollama non è in esecuzione. Assicurati che Ollama sia avviato sul tuo sistema."
        
        # Usa il dataset completo direttamente dal CSV
        complete_data = load_data()
        
        # Cerca prima se viene menzionato un medicinale specifico per nome
        prompt_lower = prompt.lower()
        mentioned_medicine = None
        
        # Cerca il nome del medicinale nella domanda
        for _, med in complete_data.iterrows():
            med_name = str(med['Medicine Name']).lower()
            if med_name in prompt_lower:
                mentioned_medicine = med
                break
        
        # Se viene trovato un medicinale specifico per nome
        if mentioned_medicine is not None:
            system_prompt = """Sei un assistente medico esperto che risponde in italiano.
Fornisci informazioni sul medicinale richiesto, ricordando sempre di:
1. Descrivere gli usi generali del medicinale
2. NON fare prescrizioni o suggerire dosaggi
3. Suggerire SEMPRE di consultare un medico per prescrizioni e dosaggi
4. Menzionare l'importanza di seguire le indicazioni del foglietto illustrativo
5. Ricordare che solo un medico può prescrivere e modificare le terapie"""

            context = {
                'nome': mentioned_medicine['Medicine Name'],
                'utilizzi': mentioned_medicine.get('Uses_IT', mentioned_medicine.get('Uses', '')),
                'effetti_collaterali': mentioned_medicine.get('Side_effects_IT', mentioned_medicine.get('Side_effects', '')),
                'produttore': mentioned_medicine['Manufacturer']
            }

            full_prompt = f"""Domanda dell'utente sul medicinale: {prompt}

Informazioni disponibili sul medicinale:
{json.dumps(context, ensure_ascii=False)}

Per favore:
1. Descrivi gli utilizzi generali del medicinale
2. NON fornire indicazioni su dosaggi o modalità di assunzione
3. Ricorda che solo un medico può prescrivere questo medicinale
4. Suggerisci SEMPRE di consultare un medico per qualsiasi utilizzo
5. Mantieni un tono professionale e cauto"""

            return generate_ollama_response(full_prompt, system_prompt)
        
        # Ricerca semplificata per sintomi
        relevant_medicines = []
        
        # Parole chiave di base per il dolore
        dolor_keywords = ['dolore', 'dolor', 'pain', 'male', 'ginocchio', 'knee', 'schiena', 'back', 'testa', 'head', 'muscolo', 'muscle', 'articolazione', 'joint']
        
        # Scorri tutti i medicinali nel dataset
        for _, med in complete_data.iterrows():
            relevance_score = 0
            
            # Preparazione testi per la ricerca
            med_name = str(med['Medicine Name']).lower()
            med_uses = str(med.get('Uses', '')).lower()
            med_composition = str(med.get('Composition', '')).lower()
            
            # Ricerca semplice ed efficace
            # 1. Se la query contiene "dolore" o "pain", cerca "pain" negli usi
            if any(kw in prompt_lower for kw in ['dolore', 'dolor', 'pain', 'male']):
                if 'pain' in med_uses or 'relief' in med_uses:
                    relevance_score += 10
            
            # 2. Ricerca per parti del corpo specifiche
            if 'ginocchio' in prompt_lower or 'knee' in prompt_lower:
                if 'joint' in med_uses or 'muscle' in med_uses or 'pain' in med_uses:
                    relevance_score += 8
            
            if 'schiena' in prompt_lower or 'back' in prompt_lower:
                if 'muscle' in med_uses or 'pain' in med_uses:
                    relevance_score += 8
            
            if 'testa' in prompt_lower or 'head' in prompt_lower or 'mal di testa' in prompt_lower:
                if 'headache' in med_uses or 'head' in med_uses:
                    relevance_score += 8
            
            # 3. Bonus per principi attivi comuni
            common_painkillers = ['ibuprofen', 'paracetamol', 'diclofenac', 'aspirin']
            if any(ingredient in med_composition for ingredient in common_painkillers):
                if any(kw in prompt_lower for kw in ['dolore', 'dolor', 'pain', 'male']):
                    relevance_score += 5
            
            # Aggiungi alla lista se ha rilevanza
            if relevance_score > 0:
                relevant_medicines.append({
                    'medicine': med,
                    'score': relevance_score,
                    'name': med['Medicine Name'],
                    'uses': med.get('Uses', ''),
                    'composition': med.get('Composition', ''),
                    'manufacturer': med.get('Manufacturer', '')
                })
        
        # Ordina per rilevanza
        relevant_medicines.sort(key=lambda x: x['score'], reverse=True)
        
        # Prendi i migliori risultati
        top_medicines = relevant_medicines[:5]
        
        if top_medicines:
            system_prompt = """Sei un assistente medico esperto che risponde in italiano.
Fornisci informazioni sui medicinali trovati che potrebbero essere rilevanti per i sintomi descritti.
IMPORTANTE:
1. NON prescrivere mai medicinali specifici o dosaggi
2. Spiega che solo un medico può prescrivere medicinali
3. Descrivi le categorie generali di trattamenti disponibili
4. Enfatizza sempre l'importanza della consultazione medica
5. Mantieni un tono professionale e cauto"""
            
            # Prepara il contesto
            context = {
                'domanda_utente': prompt,
                'medicinali_trovati': [
                    {
                        'nome': med['name'],
                        'utilizzi': med['uses'],
                        'composizione': med['composition'],
                        'produttore': med['manufacturer'],
                        'punteggio_rilevanza': med['score']
                    }
                    for med in top_medicines
                ]
            }
            
            full_prompt = f"""Domanda dell'utente: {prompt}

Medicinali rilevanti trovati nel database:
{json.dumps(context, ensure_ascii=False)}

Per favore:
1. Riconosci i sintomi descritti dall'utente (ad esempio: dolore al ginocchio)
2. Spiega che nel database sono presenti questi medicinali che sono generalmente utilizzati per il sollievo dal dolore
3. Menziona i nomi dei medicinali trovati: {', '.join([med['name'] for med in top_medicines])}
4. SOTTOLINEA CHIARAMENTE che solo un medico può prescrivere il trattamento appropriato
5. Suggerisci di consultare un professionista sanitario per una valutazione e prescrizione corretta
6. Ricorda l'importanza di non auto-medicarsi
7. Menziona che esistono diverse opzioni di trattamento disponibili"""
            
            return generate_ollama_response(full_prompt, system_prompt)
        
        # Se non trova medicinali rilevanti
        return """Mi dispiace, ma non ho trovato medicinali specifici nel database che corrispondano ai sintomi che hai descritto.

Tuttavia, ti suggerisco di:
1. Consultare il tuo medico di base per una valutazione professionale
2. In caso di dolore intenso, rivolgerti al pronto soccorso
3. Consultare un farmacista per consigli generali sui trattamenti disponibili

Ricorda che solo un medico può prescrivere il trattamento più appropriato per la tua situazione specifica."""
            
    except Exception as e:
        return f"Mi dispiace, si è verificato un errore durante la ricerca: {str(e)}"

# ============= INIZIALIZZAZIONE STATO CHATBOT =============
# Inizializzazione della cronologia dei messaggi
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============= LAYOUT PRINCIPALE =============
st.subheader(get_text("medicine_list"))

# Preparazione del DataFrame per la visualizzazione
if st.session_state.language == "Italiano":
    display_df = filtered_df[['Medicine Name', 'Composition', 'Uses_IT', 'Side_effects_IT']].copy()
    display_df.columns = [get_text("medicine_name"), get_text("composition"), get_text("uses"), get_text("side_effects")]
else:
    display_df = filtered_df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].copy()
    display_df.columns = [get_text("medicine_name"), get_text("composition"), get_text("uses"), get_text("side_effects")]

# Visualizzazione del DataFrame con gestione dello stato di disabilitazione
display_container = st.container()
with display_container:
    if not st.session_state.chatbot_processing:
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.markdown('<div class="disabled-element">', unsafe_allow_html=True)
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ============= LAYOUT STATISTICHE E CHATBOT =============
# Creazione di due colonne per il layout
col1, col2 = st.columns([1, 1])

# Colonna delle statistiche
with col1:
    if not st.session_state.chatbot_processing:
        st.subheader(get_text("statistics"))
        
        # Preparazione e visualizzazione del grafico delle recensioni
        avg_reviews = pd.DataFrame({
            'Tipo': [get_text("excellent"), get_text("average"), get_text("poor")],
            'Percentuale': [
                filtered_df['Excellent Review %'].mean(),
                filtered_df['Average Review %'].mean(),
                filtered_df['Poor Review %'].mean()
            ]
        })
        
        # Creazione del grafico a torta
        fig_reviews = px.pie(
            avg_reviews,
            values='Percentuale',
            names='Tipo',
            title=get_text("reviews_distribution")
        )
        st.plotly_chart(fig_reviews, use_container_width=True)
        
        # ============= PANORAMICA GENERALE =============
        st.subheader(get_text("overview"))
        
        # Calcolo delle statistiche
        total_medicines = len(filtered_df)
        unique_manufacturers = filtered_df['Manufacturer'].nunique()
        avg_excellent = filtered_df['Excellent Review %'].mean()
        
        # Trova il miglior medicinale (quello con più recensioni eccellenti)
        best_medicine_idx = filtered_df['Excellent Review %'].idxmax()
        best_medicine_name = filtered_df.loc[best_medicine_idx, 'Medicine Name']
        best_medicine_score = filtered_df.loc[best_medicine_idx, 'Excellent Review %']
        
        # Layout a 2x2 per le metriche
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label=get_text("total_medicines"),
                value=f"{total_medicines:,}",
                delta=f"↑ di {total_medicines} totali"
            )
            
            st.metric(
                label=get_text("avg_excellent_reviews"),
                value=f"{avg_excellent:.1f}%",
                delta="↑ recensioni positive"
            )
        
        with metric_col2:
            st.metric(
                label=get_text("unique_manufacturers"),
                value=f"{unique_manufacturers}",
                delta="nel dataset"
            )
            
            st.metric(
                label=get_text("best_medicine"),
                value=best_medicine_name[:20] + "..." if len(best_medicine_name) > 20 else best_medicine_name,
                delta=f"{best_medicine_score:.1f}% eccellenti"
            )
    else:
        st.markdown('<div class="disabled-element">', unsafe_allow_html=True)
        st.subheader(get_text("statistics"))
        
        # Preparazione e visualizzazione del grafico delle recensioni
        avg_reviews = pd.DataFrame({
            'Tipo': [get_text("excellent"), get_text("average"), get_text("poor")],
            'Percentuale': [
                filtered_df['Excellent Review %'].mean(),
                filtered_df['Average Review %'].mean(),
                filtered_df['Poor Review %'].mean()
            ]
        })
        
        # Creazione del grafico a torta
        fig_reviews = px.pie(
            avg_reviews,
            values='Percentuale',
            names='Tipo',
            title=get_text("reviews_distribution")
        )
        st.plotly_chart(fig_reviews, use_container_width=True)
        
        # ============= PANORAMICA GENERALE =============
        st.subheader(get_text("overview"))
        
        # Calcolo delle statistiche
        total_medicines = len(filtered_df)
        unique_manufacturers = filtered_df['Manufacturer'].nunique()
        avg_excellent = filtered_df['Excellent Review %'].mean()
        
        # Trova il miglior medicinale (quello con più recensioni eccellenti)
        best_medicine_idx = filtered_df['Excellent Review %'].idxmax()
        best_medicine_name = filtered_df.loc[best_medicine_idx, 'Medicine Name']
        best_medicine_score = filtered_df.loc[best_medicine_idx, 'Excellent Review %']
        
        # Layout a 2x2 per le metriche
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label=get_text("total_medicines"),
                value=f"{total_medicines:,}",
                delta=f"↑ di {total_medicines} totali"
            )
            
            st.metric(
                label=get_text("avg_excellent_reviews"),
                value=f"{avg_excellent:.1f}%",
                delta="↑ recensioni positive"
            )
        
        with metric_col2:
            st.metric(
                label=get_text("unique_manufacturers"),
                value=f"{unique_manufacturers}",
                delta="nel dataset"
            )
            
            st.metric(
                label=get_text("best_medicine"),
                value=best_medicine_name[:20] + "..." if len(best_medicine_name) > 20 else best_medicine_name,
                delta=f"{best_medicine_score:.1f}% eccellenti"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# Colonna del chatbot
with col2:
    # CSS per layout e altezza colonna
    st.markdown("""
    <style>
    .chatbot-column {
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .chat-content {
        flex: 1;
        overflow-y: auto;
    }
    .chat-input-container {
        margin-top: auto;
        padding-top: 1rem;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.2rem 0.5rem !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: none !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:disabled {
        opacity: 0.5 !important;
        background-color: transparent !important;
    }
    
    /* Forza i messaggi della chat nell'area corretta */
    .stChatMessage {
        margin: 0.5rem 0 !important;
    }
    
    /* Container chat personalizzato */
    .custom-chat-container {
        height: 60vh;
        overflow-y: auto;
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header con titolo e cestino affiancati
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col1:
        st.subheader(get_text("virtual_assistant"))
    
    with header_col2:
        # CSS specifico per rendere il pulsante più piccolo e spostato a sinistra
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: transparent !important;
            border: none !important;
            color: #ffffff !important;
            font-size: 1rem !important;
            padding: 0.1rem 0.3rem !important;
            margin: 0 !important;
            box-shadow: none !important;
            width: 2rem !important;
            height: 2rem !important;
            position: relative;
            left: -2.5rem;
            top: 0.8rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if not st.session_state.chatbot_processing:
            if st.button("🗑️", key="clear_chat_header", help="Pulisci Chat", 
                        type="secondary", use_container_width=False):
                st.session_state.messages = []
                st.session_state.response_cache = {}
                st.rerun()
        else:
            st.button("🗑️", disabled=True, key="clear_chat_header_disabled", 
                     help="Pulisci Chat", type="secondary", use_container_width=False)
    
    # Area chat con container Streamlit nativo
    with st.container():
        # CSS per forzare il container nell'area corretta
        st.markdown("""
        <style>
        /* Forza il container della chat in un'area specifica */
        .stContainer > div {
            height: 60vh !important;
            overflow-y: auto !important;
            border: 2px solid #444 !important;
            border-radius: 0.5rem !important;
            padding: 1rem !important;
            background-color: rgba(0,0,0,0.3) !important;
            margin-bottom: 1rem !important;
        }
        
        /* Stile per i messaggi della chat */
        .stChatMessage {
            margin: 0.5rem 0 !important;
            padding: 0.5rem !important;
        }
        
        /* Forza i messaggi a rimanere nel container */
        [data-testid="stChatMessage"] {
            position: relative !important;
            z-index: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Se non ci sono messaggi, mostra un messaggio di benvenuto
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; color: #888; margin-top: 2rem;">
                <p>👋 Ciao! Sono il tuo assistente virtuale per i medicinali.</p>
                <p>Fai una domanda qui sotto per iniziare!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Visualizzazione messaggi esistenti con componenti nativi
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Elaborazione della risposta se il chatbot è in stato di processing
        if st.session_state.chatbot_processing and len(st.session_state.messages) > 0:
            # Mostra indicatore di elaborazione
            with st.chat_message("assistant"):
                st.markdown("🔄 Sto elaborando la tua richiesta...")
            
            # Prendi l'ultimo messaggio utente
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user":
                # Preparazione del contesto con i dati dei medicinali
                medicine_data = filtered_df.to_dict('records')
                
                # Generazione della risposta
                response = get_local_ai_response(last_message["content"], medicine_data)
                
                # Aggiunta della risposta alla cronologia
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Ripristina lo stato di elaborazione
                st.session_state.chatbot_processing = False
                st.rerun()  # Aggiorna l'interfaccia per riabilitare gli elementi
    
    # JavaScript per forzare i messaggi nel container
    st.markdown("""
    <script>
    // Funzione per spostare i messaggi nel container
    function moveMessagesToContainer() {
        setTimeout(function() {
            // Trova tutti i messaggi di chat
            const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
            // Trova il container della chat (il primo stContainer nella colonna)
            const chatContainer = document.querySelector('.stContainer');
            
            if (chatContainer && chatMessages.length > 0) {
                console.log('Trovati', chatMessages.length, 'messaggi da spostare');
                
                // Crea un div interno per i messaggi se non esiste
                let messagesDiv = chatContainer.querySelector('.chat-messages-area');
                if (!messagesDiv) {
                    messagesDiv = document.createElement('div');
                    messagesDiv.className = 'chat-messages-area';
                    messagesDiv.style.cssText = `
                        height: 100%;
                        overflow-y: auto;
                        padding: 0.5rem;
                    `;
                    chatContainer.appendChild(messagesDiv);
                }
                
                // Sposta ogni messaggio nel container
                chatMessages.forEach(function(message, index) {
                    if (message.parentNode !== messagesDiv) {
                        console.log('Spostando messaggio', index);
                        messagesDiv.appendChild(message);
                    }
                });
                
                // Scorri verso il basso per mostrare l'ultimo messaggio
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        }, 500); // Aspetta che Streamlit finisca il rendering
    }
    
    // Esegui la funzione
    moveMessagesToContainer();
    
    // Ripeti ogni secondo per catturare nuovi messaggi
    setInterval(moveMessagesToContainer, 1000);
    </script>
    """, unsafe_allow_html=True)
    
    # Input fisso in fondo alla colonna
    if prompt := st.chat_input(get_text("ask_question"), disabled=st.session_state.chatbot_processing):
        # Aggiunta del messaggio dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Imposta lo stato di elaborazione
        st.session_state.chatbot_processing = True
        
        # Aggiorna per mostrare il messaggio utente e l'indicatore di elaborazione
        st.rerun()

# ============= DETTAGLI MEDICINALE =============
st.markdown("---")
st.subheader(get_text("medicine_details"))

# Selezione del medicinale con gestione dello stato di disabilitazione
if not st.session_state.chatbot_processing:
    selected_medicine = st.selectbox(
        get_text("select_medicine"),
        filtered_df['Medicine Name'].unique()
    )
else:
    st.markdown('<div class="disabled-element">', unsafe_allow_html=True)
    selected_medicine = st.selectbox(
        get_text("select_medicine"),
        filtered_df['Medicine Name'].unique(),
        disabled=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizzazione dei dettagli del medicinale selezionato
if selected_medicine:
    medicine_details = filtered_df[filtered_df['Medicine Name'] == selected_medicine].iloc[0]
    
    if not st.session_state.chatbot_processing:
        # Layout a tre colonne per i dettagli
        col3, col4, col5 = st.columns(3)
        
        # Prima colonna: Nome e composizione
        with col3:
            st.metric(get_text("name"), medicine_details['Medicine Name'])
            st.metric(get_text("composition"), medicine_details['Composition'])
        
        # Seconda colonna: Recensioni eccellenti e nella media
        with col4:
            st.metric(get_text("excellent_reviews"), f"{medicine_details['Excellent Review %']}%")
            st.metric(get_text("average_reviews"), f"{medicine_details['Average Review %']}%")
        
        # Terza colonna: Recensioni scarse e produttore
        with col5:
            st.metric(get_text("poor_reviews"), f"{medicine_details['Poor Review %']}%")
            st.metric(get_text("manufacturer"), medicine_details['Manufacturer'])
        
        # Visualizzazione degli utilizzi
        st.markdown("### " + get_text("uses"))
        if st.session_state.language == "Italiano":
            st.write(medicine_details['Uses_IT'])
        else:
            st.write(medicine_details['Uses'])
        
        # Visualizzazione degli effetti collaterali
        st.markdown("### " + get_text("side_effects"))
        if st.session_state.language == "Italiano":
            st.write(medicine_details['Side_effects_IT'])
        else:
            st.write(medicine_details['Side_effects'])
        
        # Visualizzazione dell'immagine del medicinale se disponibile
        if pd.notna(medicine_details['Image URL']):
            st.image(medicine_details['Image URL'], caption=medicine_details['Medicine Name'])
    else:
        st.markdown('<div class="disabled-element">', unsafe_allow_html=True)
        # Layout a tre colonne per i dettagli
        col3, col4, col5 = st.columns(3)
        
        # Prima colonna: Nome e composizione
        with col3:
            st.metric(get_text("name"), medicine_details['Medicine Name'])
            st.metric(get_text("composition"), medicine_details['Composition'])
        
        # Seconda colonna: Recensioni eccellenti e nella media
        with col4:
            st.metric(get_text("excellent_reviews"), f"{medicine_details['Excellent Review %']}%")
            st.metric(get_text("average_reviews"), f"{medicine_details['Average Review %']}%")
        
        # Terza colonna: Recensioni scarse e produttore
        with col5:
            st.metric(get_text("poor_reviews"), f"{medicine_details['Poor Review %']}%")
            st.metric(get_text("manufacturer"), medicine_details['Manufacturer'])
        
        # Visualizzazione degli utilizzi
        st.markdown("### " + get_text("uses"))
        if st.session_state.language == "Italiano":
            st.write(medicine_details['Uses_IT'])
        else:
            st.write(medicine_details['Uses'])
        
        # Visualizzazione degli effetti collaterali
        st.markdown("### " + get_text("side_effects"))
        if st.session_state.language == "Italiano":
            st.write(medicine_details['Side_effects_IT'])
        else:
            st.write(medicine_details['Side_effects'])
        
        # Visualizzazione dell'immagine del medicinale se disponibile
        if pd.notna(medicine_details['Image URL']):
            st.image(medicine_details['Image URL'], caption=medicine_details['Medicine Name'])
        st.markdown('</div>', unsafe_allow_html=True)
