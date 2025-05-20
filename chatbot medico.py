# Importazioni di base
import streamlit as st
import sys
import pandas as pd
import plotly.express as px
from deep_translator import GoogleTranslator
import time
import os
import json
import requests
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import threading
import asyncio
import nest_asyncio
from pydantic import BaseModel
import gc
import socket

# Configurazione della pagina Streamlit (DEVE essere il primo comando Streamlit)
st.set_page_config(
    page_title="Dashboard Medicinali",
    page_icon="💊",
    layout="wide"
)

# Configurazione dell'event loop per FastAPI e Streamlit
async def setup_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply()

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

# Inizializzazione FastAPI
app = FastAPI()

def run_fastapi():
    try:
        free_port = find_free_port()
        if not free_port:
            st.warning("Non è stato possibile trovare una porta libera per il server FastAPI. L'applicazione continuerà a funzionare ma alcune funzionalità potrebbero essere limitate.")
            return

        config = uvicorn.Config(app, host="127.0.0.1", port=free_port, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(setup_async())
        try:
            asyncio.run(server.serve())
        except SystemExit:
            pass  # Ignora l'uscita del sistema quando il server viene fermato
        except Exception as e:
            st.error(f"Errore durante l'avvio del server FastAPI: {str(e)}")
    except Exception as e:
        st.error(f"Errore nella configurazione del server: {str(e)}")

# Avvio del server FastAPI in un thread separato
if 'fastapi_thread' not in st.session_state:
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    st.session_state.fastapi_thread = fastapi_thread
    time.sleep(1)  # Breve pausa per permettere l'avvio del server

# Stile CSS per il toggle della lingua
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

# Titolo principale
st.title("💊 Dashboard Consultazione Medicinali")
st.markdown("---")

# Sidebar - Toggle della lingua in alto
is_italian = st.sidebar.toggle("🌍 EN / IT", value=True, key="language_toggle")
st.session_state.language = "Italiano" if is_italian else "English"

# Sidebar - Filtri
st.sidebar.header("Filtri")

# Ricerca per nome
search_term = st.sidebar.text_input("Cerca medicinale per nome:")

# Configurazione Ollama
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

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

# Funzione per verificare lo stato dell'API
def check_api_status():
    try:
        # Test semplice per verificare la connessione
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test connection"}],
            temperature=0.7,
            max_tokens=500,
            stream=False
        )
        return True
    except Exception as e:
        st.error(f"Errore di connessione all'API: {str(e)}")
        return False

# Classe per la richiesta API
class ChatRequest(BaseModel):
    prompt: str
    context: str

# Configurazione del modello
MODEL_NAME = "mistral"
DEFAULT_SYSTEM_PROMPT = """Sei un assistente medico esperto che risponde SEMPRE in italiano. 
Il tuo compito è aiutare gli utenti fornendo informazioni accurate sui medicinali basandoti sui dati forniti. 
Rispondi in modo chiaro e professionale, utilizzando ESCLUSIVAMENTE la lingua italiana."""

# Caricamento del modello (verrà fatto una sola volta)
@st.cache_resource
def load_llm():
    try:
        # Configurazione del tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # Configurazione del modello ottimizzata per CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        
        # Forza il modello in modalità valutazione per risparmiare memoria
        model.eval()
        
        # Creazione della pipeline ottimizzata
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            return_full_text=False,
            batch_size=1
        )
        
        return pipe
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {str(e)}")
        return None

# Funzione per generare risposte
def generate_response(pipe, prompt, context):
    if pipe is None:
        return "Errore: Modello non caricato correttamente. Verifica il login a Hugging Face."
    
    try:
        # Crea il prompt completo
        full_prompt = f"""<|system|>
{DEFAULT_SYSTEM_PROMPT}

Contesto sui medicinali:
{context}</|system|>
<|user|>
{prompt}</|user|>
<|assistant|>"""
        
        # Genera la risposta
        response = pipe(
            full_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            return_full_text=False
        )[0]["generated_text"]
        
        # Pulisci la risposta
        response = response.split("</|assistant|>")[0].strip()
        return response
    except Exception as e:
        return f"Mi dispiace, si è verificato un errore nella generazione della risposta: {str(e)}"

# Endpoint FastAPI
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    model, tokenizer = load_llm()
    if model is None or tokenizer is None:
        return {"response": "Errore: Modello non caricato correttamente"}
    
    try:
        # Formatta il prompt
        full_prompt = f"""Sei un assistente medico esperto. Basandoti sui dati forniti, rispondi alla domanda dell'utente.

### Contesto (informazioni sui medicinali):
{request.context}

### Domanda dell'utente:
{request.prompt}

### Risposta:"""
        
        # Genera la risposta
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Estrai solo la parte della risposta dopo "### Risposta:"
        if "### Risposta:" in response:
            response = response.split("### Risposta:")[-1].strip()
        
        return {"response": response}
    except Exception as e:
        return {"response": f"Errore nella generazione della risposta: {str(e)}"}

# Funzione per verificare se il server è in esecuzione
def is_server_running():
    try:
        response = requests.get("http://127.0.0.1:8000/docs")
        return response.status_code == 200
    except:
        return False

# Funzione per avviare il server FastAPI in background
def run_api_server():
    try:
        nest_asyncio.apply()
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        st.error(f"Errore nell'avvio del server: {str(e)}")

# Avvia il server FastAPI in un thread separato se non è già in esecuzione
if not is_server_running():
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Attendi che il server si avvii

# Cache per le risposte del chatbot
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

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

# Funzione per generare risposte con Ollama
def generate_ollama_response(prompt, system_prompt, max_tokens=500):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,  # Disabilitiamo lo streaming per evitare la doppia stampa
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        # Prova prima a ottenere una risposta dalla cache
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

# Funzione per verificare se Ollama è in esecuzione
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def get_local_ai_response(prompt, medicine_data):
    try:
        if not check_ollama_status():
            return "Errore: Ollama non è in esecuzione. Assicurati che Ollama sia avviato sul tuo sistema."
        
        # Analisi semantica della domanda
        query_terms = set(prompt.lower().split())
        query_type = {
            'sintomi': any(term in query_terms for term in ['sintomo', 'sintomi', 'male', 'dolore', 'disturbo']),
            'effetti_collaterali': any(term in query_terms for term in ['effetto', 'collaterale', 'reazione']),
            'dosaggio': any(term in query_terms for term in ['dose', 'dosaggio', 'quanto', 'assumere', 'assunzione']),
            'controindicazioni': any(term in query_terms for term in ['controindicazione', 'controindicato', 'evitare']),
            'generale': any(term in query_terms for term in ['cosa', 'come', 'quando', 'perché'])
        }
        
        # Ricerca avanzata nei medicinali
        relevant_meds = []
        partial_matches = []
        
        for med in medicine_data:
            med_name = med.get('Medicine Name', '').lower()
            med_uses = med.get('Uses_IT', '').lower()
            med_effects = med.get('Side_effects_IT', '').lower()
            med_composition = med.get('Composition', '').lower()
            
            # Calcolo diversi tipi di rilevanza
            name_relevance = sum(1 for term in query_terms if term in med_name) * 2
            uses_relevance = sum(1 for term in query_terms if term in med_uses)
            effects_relevance = sum(1 for term in query_terms if term in med_effects)
            composition_relevance = sum(1 for term in query_terms if term in med_composition)
            
            total_relevance = name_relevance + uses_relevance + effects_relevance + composition_relevance
            
            if total_relevance > 0:
                relevant_meds.append((total_relevance, med))
            elif any(term in (med_uses + med_effects) for term in query_terms):
                partial_matches.append((0.5, med))
        
        # Combina e ordina i risultati
        all_matches = sorted(relevant_meds + partial_matches, key=lambda x: x[0], reverse=True)
        
        system_prompt = """Sei un assistente medico esperto che risponde in italiano.
Fornisci informazioni accurate sui medicinali basandoti sui dati forniti.
Se non hai informazioni specifiche dal dataset:
1. Indica chiaramente che stai fornendo informazioni generali
2. Suggerisci alternative o approcci correlati
3. Mantieni un tono professionale e cauto
4. Se appropriato, suggerisci di consultare un medico"""
        
        if not all_matches:
            # Analizza il tipo di domanda per fornire una risposta pertinente
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
            # Preparazione ottimizzata del contesto con informazioni complete
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

# Inizializzazione della sessione per il chatbot
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Layout principale
st.subheader("Lista Medicinali")
display_df = filtered_df[['Medicine Name', 'Composition', 'Uses_IT', 'Side_effects_IT']].copy()
display_df.columns = ['Nome Medicinale', 'Composizione', 'Utilizzi', 'Effetti Collaterali']
if st.session_state.language == "English":
    display_df = filtered_df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].copy()
    display_df.columns = ['Medicine Name', 'Composition', 'Uses', 'Side Effects']

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)

# Layout a due colonne per statistiche e chatbot
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Statistiche")
    
    # Grafico delle recensioni
    avg_reviews = pd.DataFrame({
        'Tipo': ['Eccellenti', 'Nella Media', 'Scarse'],
        'Percentuale': [
            filtered_df['Excellent Review %'].mean(),
            filtered_df['Average Review %'].mean(),
            filtered_df['Poor Review %'].mean()
        ]
    })
    
    fig_reviews = px.pie(
        avg_reviews,
        values='Percentuale',
        names='Tipo',
        title='Distribuzione Media delle Recensioni'
    )
    st.plotly_chart(fig_reviews, use_container_width=True)

with col2:
    st.subheader("💬 Assistente Virtuale (Mistral)")
    
    # Mostra la cronologia dei messaggi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input per il chatbot
    if prompt := st.chat_input("Fai una domanda sui medicinali..."):
        # Aggiungi il messaggio dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostra il messaggio dell'utente
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepara i dati dei medicinali per il contesto
        medicine_data = filtered_df.to_dict('records')
        
        # Genera e mostra la risposta
        with st.chat_message("assistant"):
            with st.spinner("Elaborazione in corso..."):
                response = get_local_ai_response(prompt, medicine_data)
            st.markdown(response)
        
        # Aggiungi la risposta alla cronologia
        st.session_state.messages.append({"role": "assistant", "content": response})

# Dettagli medicinale
st.markdown("---")
st.subheader("Dettagli Medicinale")
selected_medicine = st.selectbox(
    "Seleziona un medicinale per vedere i dettagli:",
    filtered_df['Medicine Name'].unique()
)

if selected_medicine:
    medicine_details = filtered_df[filtered_df['Medicine Name'] == selected_medicine].iloc[0]
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Nome" if st.session_state.language == "Italiano" else "Name", medicine_details['Medicine Name'])
        st.metric("Composizione" if st.session_state.language == "Italiano" else "Composition", medicine_details['Composition'])
    
    with col4:
        st.metric("Recensioni Eccellenti" if st.session_state.language == "Italiano" else "Excellent Reviews", f"{medicine_details['Excellent Review %']}%")
        st.metric("Recensioni Nella Media" if st.session_state.language == "Italiano" else "Average Reviews", f"{medicine_details['Average Review %']}%")
    
    with col5:
        st.metric("Recensioni Scarse" if st.session_state.language == "Italiano" else "Poor Reviews", f"{medicine_details['Poor Review %']}%")
        st.metric("Produttore" if st.session_state.language == "Italiano" else "Manufacturer", medicine_details['Manufacturer'])
    
    st.markdown("### " + ("Utilizzi" if st.session_state.language == "Italiano" else "Uses"))
    if st.session_state.language == "Italiano":
        st.write(medicine_details['Uses_IT'])
    else:
        st.write(medicine_details['Uses'])
    
    st.markdown("### " + ("Effetti Collaterali" if st.session_state.language == "Italiano" else "Side Effects"))
    if st.session_state.language == "Italiano":
        st.write(medicine_details['Side_effects_IT'])
    else:
        st.write(medicine_details['Side_effects'])
    
    if pd.notna(medicine_details['Image URL']):
        st.image(medicine_details['Image URL'], caption=medicine_details['Medicine Name'])
