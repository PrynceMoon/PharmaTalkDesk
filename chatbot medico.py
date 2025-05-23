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

# ============= CONFIGURAZIONE STREAMLIT =============
# Impostazione della configurazione della pagina web
st.set_page_config(
    page_title="Dashboard Medicinali",  # Titolo della pagina
    page_icon="💊",  # Icona della pagina
    layout="wide"  # Layout a schermo intero
)

# ============= CONFIGURAZIONE ASINCRONA =============
# Funzione per configurare l'event loop asincrono
async def setup_async():
    try:
        loop = asyncio.get_event_loop()  # Prova a ottenere l'event loop esistente
    except RuntimeError:
        loop = asyncio.new_event_loop()  # Crea un nuovo event loop se non esiste
        asyncio.set_event_loop(loop)
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
        
        # Configura e avvia il server
        config = uvicorn.Config(app, host="127.0.0.1", port=fastapi_port, log_level="info")
        server = uvicorn.Server(config)
        
        try:
            asyncio.run(setup_async())  # Configura l'ambiente asincrono
            asyncio.run(server.serve())  # Avvia il server
        except SystemExit:
            pass  # Gestisce l'uscita normale del server
        except Exception as e:
            st.error(f"Errore durante l'esecuzione del server FastAPI: {str(e)}")
    except Exception as e:
        st.error(f"Errore nella configurazione del server: {str(e)}")

# Avvio del server FastAPI in un thread separato
if 'fastapi_thread' not in st.session_state:
    if 'fastapi_thread' in st.session_state and st.session_state.fastapi_thread.is_alive():
        pass  # Non crea un nuovo thread se ne esiste già uno attivo
    else:
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_thread = fastapi_thread
        time.sleep(1)  # Pausa per permettere l'avvio del server

# ============= STILE CSS =============
# Definizione dello stile per il pulsante di cambio lingua
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #f0f2f6;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============= GESTIONE DELLA LINGUA =============
# Inizializzazione della lingua predefinita
if 'language' not in st.session_state:
    st.session_state.language = "Italiano"

# Pulsante per il cambio della lingua nella sidebar
is_italian = st.sidebar.toggle("🌍 EN / IT", value=True, key="language_toggle")
st.session_state.language = "Italiano" if is_italian else "English"

# ============= GESTIONE DEL TESTO MULTILINGUA =============
def get_text(key):
    """Restituisce il testo nella lingua selezionata"""
    # Dizionario delle traduzioni per tutte le stringhe dell'interfaccia
    translations = {
        "page_title": {
            "it": "Dashboard Consultazione Medicinali",
            "en": "Medicines Consultation Dashboard"
        },
        "filters": {
            "it": "Filtri",
            "en": "Filters"
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
    }
    
    # Determina la lingua corrente e restituisce la traduzione appropriata
    lang = "it" if st.session_state.language == "Italiano" else "en"
    return translations.get(key, {}).get(lang, key)

# ============= TITOLO PRINCIPALE =============
st.title("💊 " + get_text("page_title"))
st.markdown("---")

# ============= CONFIGURAZIONE SIDEBAR E FILTRI =============
# Aggiunta dell'header dei filtri nella sidebar
st.sidebar.header(get_text("filters"))

# Campo di ricerca per nome del medicinale
search_term = st.sidebar.text_input(get_text("search_medicine"))

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
            st.session_state.response_cache[prompt] = result
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
        
        # Analisi semantica della domanda dell'utente
        query_terms = set(prompt.lower().split())
        query_type = {
            'sintomi': any(term in query_terms for term in ['sintomo', 'sintomi', 'male', 'dolore', 'disturbo']),
            'effetti_collaterali': any(term in query_terms for term in ['effetto', 'collaterale', 'reazione']),
            'dosaggio': any(term in query_terms for term in ['dose', 'dosaggio', 'quanto', 'assumere', 'assunzione']),
            'controindicazioni': any(term in query_terms for term in ['controindicazione', 'controindicato', 'evitare']),
            'generale': any(term in query_terms for term in ['cosa', 'come', 'quando', 'perché'])
        }
        
        # Ricerca nei dati dei medicinali
        relevant_meds = []
        partial_matches = []
        
        # Analisi di ogni medicinale nel database
        for med in medicine_data:
            med_name = med.get('Medicine Name', '').lower()
            med_uses = med.get('Uses_IT', '').lower()
            med_effects = med.get('Side_effects_IT', '').lower()
            med_composition = med.get('Composition', '').lower()
            
            # Calcolo della rilevanza del medicinale per la query
            name_relevance = sum(1 for term in query_terms if term in med_name) * 2
            uses_relevance = sum(1 for term in query_terms if term in med_uses)
            effects_relevance = sum(1 for term in query_terms if term in med_effects)
            composition_relevance = sum(1 for term in query_terms if term in med_composition)
            
            total_relevance = name_relevance + uses_relevance + effects_relevance + composition_relevance
            
            # Classificazione dei risultati
            if total_relevance > 0:
                relevant_meds.append((total_relevance, med))
            elif any(term in (med_uses + med_effects) for term in query_terms):
                partial_matches.append((0.5, med))
        
        # Ordinamento dei risultati per rilevanza
        all_matches = sorted(relevant_meds + partial_matches, key=lambda x: x[0], reverse=True)
        
        # Configurazione del prompt di sistema per la risposta
        system_prompt = """Sei un assistente medico esperto che risponde in italiano.
Fornisci informazioni accurate sui medicinali basandoti sui dati forniti.
Se non hai informazioni specifiche dal dataset:
1. Indica chiaramente che stai fornendo informazioni generali
2. Suggerisci alternative o approcci correlati
3. Mantieni un tono professionale e cauto
4. Se appropriato, suggerisci di consultare un medico"""

        # Generazione della risposta in base ai risultati trovati
        if not all_matches:
            # Gestione caso: nessun medicinale trovato
            context = {
                "tipo_richiesta": prompt,
                "tipo_domanda": [k for k, v in query_type.items() if v],
                "medicinali_trovati": False
            }
            
            full_prompt = f"""La domanda dell'utente riguarda: {', '.join(context['tipo_domanda']) if context['tipo_domanda'] else 'informazione generale'}
Domanda specifica: {prompt}

Non ho trovato medicinali specifici nel database che corrispondano esattamente alla richiesta.
Per favore:
1. Fornisci una risposta generale pertinente al tipo di domanda
2. Se appropriato, suggerisci categorie di medicinali correlate
3. Indica chiaramente quando stai fornendo informazioni generali
4. Suggerisci di consultare un professionista sanitario se necessario
5. Proponi come riformulare la domanda per ottenere informazioni più specifiche"""
        else:
            # Preparazione del contesto con i medicinali trovati
            essential_data = [{
                'nome': med.get('Medicine Name', ''),
                'utilizzi': med.get('Uses_IT', ''),
                'effetti_collaterali': med.get('Side_effects_IT', ''),
                'composizione': med.get('Composition', ''),
                'rilevanza': score,
                'tipo_match': 'diretto' if score >= 1 else 'correlato'
            } for score, med in all_matches[:5]]
            
            context = json.dumps(essential_data, ensure_ascii=False)
            full_prompt = f"""Tipo di domanda: {', '.join([k for k, v in query_type.items() if v]) if any(query_type.values()) else 'generale'}
Dati medicinali disponibili: {context}
Domanda dell'utente: {prompt}

Per favore:
1. Rispondi in modo specifico alla domanda usando i dati forniti
2. Se i medicinali trovati sono solo parzialmente rilevanti, spiegalo chiaramente
3. Indica eventuali limitazioni nelle informazioni disponibili
4. Se necessario, suggerisci di consultare un professionista sanitario
5. Mantieni un tono professionale e accurato"""
        
        return generate_ollama_response(full_prompt, system_prompt)
            
    except Exception as e:
        st.error(f"Errore: {str(e)}")
        return f"Mi dispiace, si è verificato un errore: {str(e)}"

# ============= INIZIALIZZAZIONE STATO CHATBOT =============
# Inizializzazione della cronologia dei messaggi
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============= LAYOUT PRINCIPALE =============
st.subheader(get_text("medicine_list"))

# Preparazione del DataFrame per la visualizzazione
display_df = filtered_df[['Medicine Name', 'Composition', 'Uses_IT', 'Side_effects_IT']].copy()
display_df.columns = [get_text("medicine_name"), get_text("composition"), get_text("uses"), get_text("side_effects")]
if st.session_state.language == "English":
    display_df = filtered_df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].copy()
    display_df.columns = [get_text("medicine_name"), get_text("composition"), get_text("uses"), get_text("side_effects")]

# Visualizzazione della tabella dei medicinali
st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)

# ============= LAYOUT STATISTICHE E CHATBOT =============
# Creazione di due colonne per il layout
col1, col2 = st.columns([1, 1])

# Colonna delle statistiche
with col1:
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

# Colonna del chatbot
with col2:
    st.subheader(get_text("virtual_assistant"))
    
    # Visualizzazione della cronologia dei messaggi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input per il chatbot e gestione delle risposte
    if prompt := st.chat_input(get_text("ask_question")):
        # Aggiunta del messaggio dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Visualizzazione del messaggio dell'utente
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Preparazione del contesto con i dati dei medicinali
        medicine_data = filtered_df.to_dict('records')
        
        # Generazione e visualizzazione della risposta
        with st.chat_message("assistant"):
            with st.spinner("Elaborazione in corso..."):
                response = get_local_ai_response(prompt, medicine_data)
            st.markdown(response)
        
        # Aggiunta della risposta alla cronologia
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============= DETTAGLI MEDICINALE =============
st.markdown("---")
st.subheader(get_text("medicine_details"))

# Selezione del medicinale
selected_medicine = st.selectbox(
    get_text("select_medicine"),
    filtered_df['Medicine Name'].unique()
)

# Visualizzazione dei dettagli del medicinale selezionato
if selected_medicine:
    medicine_details = filtered_df[filtered_df['Medicine Name'] == selected_medicine].iloc[0]
    
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
