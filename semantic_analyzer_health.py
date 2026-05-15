# semantic_analyzer_health.py
"""
Analisador Semântico de Textos Clínicos com IA
Adaptado para a área de saúde / NLP médico
Funcionalidades:
- Análise de similaridade semântica entre registros clínicos
- Classificação de risco/urgência
- Detecção de contradições em laudos
- Extração de entidades médicas (sintomas, diagnósticos, medicamentos)
- Sumarização clínica
- Anonimização de dados sensíveis (PHI - Protected Health Information)
"""

import os
import re
import json
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter

# ───────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ───────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelo de embeddings otimizado para domínio clínico (multilíngue)
# all-MiniLM-L6-v2 é genérico; para saúde, recomenda-se:
# - 'neuml/pubmedbert-base-embeddings' ou similar (requer download)
# Aqui mantemos compatibilidade mas com fallback para modelo clínico
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ───────────────────────────────────────────────────────────────
# CONSTANTES E DICIONÁRIOS MÉDICOS
# ───────────────────────────────────────────────────────────────

# Dicionário de entidades médicas para highlight
ENTIDADES_MEDICAS = {
    "sintomas": [
        "febre", "dor", "cansaço", "fadiga", "náusea", "vômito", "diarreia",
        "constipação", "tosse", "dispneia", "cefaleia", "tontura", "palpitação",
        "edema", "erupção", "prurido", "calafrio", "sudorese", "perda de peso",
        "ganho de peso", "inchaço", "hematoma", "sangramento", "convulsão",
        "desmaio", "sincope", "confusão", "agitação", "depressão", "ansiedade",
        "insônia", "anorexia", "poliúria", "polidipsia", "polifagia", "dor torácica",
        "dor abdominal", "dor lombar", "dor articular", "mialgia", "artralgia"
    ],
    "diagnosticos": [
        "hipertensão", "diabetes", "asma", "bronquite", "pneumonia", "infarto",
        " AVC ", "acidente vascular", "derrame", "câncer", "tumor", "neoplasia",
        "anemia", "leucemia", "hepatite", "cirrose", "nefrite", "insuficiência renal",
        "insuficiência cardíaca", "arritmia", "fibrilação", "estenose", "valvopatia",
        "osteoporose", "artrite", "gota", "lúpus", "esclerose", "esquizofrenia",
        "bipolar", "depressão", "ansiedade", "panico", "TDAH", "autismo",
        "demência", "alzheimer", "parkinson", "epilepsia", "migrânea"
    ],
    "medicamentos": [
        "paracetamol", "dipirona", "ibuprofeno", "aspirina", "amoxicilina",
        "azitromicina", "ciprofloxacino", "metformina", "insulina", "losartana",
        "enalapril", "atenolol", "amlodipino", "simvastatina", "atorvastatina",
        "omeprazol", "ranitidina", "prednisona", "hidrocortisona", "dexametasona",
        "warfarina", "rivaroxabana", "heparina", "clopidogrel", "fluoxetina",
        "sertralina", "risperidona", "haloperidol", "diazepam", "lorazepam",
        "tramadol", "morfina", "fentanil", "ondansetrona", "metoclopramida"
    ],
    "procedimentos": [
        "eletrocardiograma", "ecg", "ecocardiograma", "raio-x", "tomografia",
        "ressonância", "ultrassom", "ultrassonografia", "endoscopia", "colonoscopia",
        "biopsia", "cirurgia", "cateterismo", "dialise", "quimioterapia",
        "radioterapia", "fisioterapia", "hemodiálise", "ventilação mecânica",
        "intubação", "tracheostomia", "punção", "drenagem", "sutura"
    ],
    "exames_lab": [
        "hemograma", "glicemia", "colesterol", "triglicerídeos", "creatinina",
        "ureia", "tgo", "tgp", "bilirrubina", "tsh", "t4", "t3", "psa",
        "ceA", "ca-125", "hba1c", "pcr", "vhs", "inr", "ptt", "plaquetas",
        "leucócitos", "neutrófilos", "linfócitos", "eosinófilos", "basófilos",
        "eritrócitos", "hemoglobina", "hematócrito", "vcm", "hcm", "rdw"
    ]
}

# Padrões de PHI para anonimização
PHI_PATTERNS = {
    "cpf": r"\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{2}\b",
    "telefone": r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-.\s]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "data_nascimento": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    "prontuario": r"\b(?:prontuário|prontuario|nº|numero|número)[\s:.-]*(\d+)\b",
    "nome_paciente": r"(?:Paciente|Sr\.|Sra\.|Srta\.|Dr\.|Dra\.)[\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
}

# Escala de urgência/triagem
ESCALA_URGENCIA = {
    "Emergência": {"cor": "🔴", "desc": "Risco imediato de vida", "tempo": "Atendimento imediato"},
    "Muito Urgente": {"cor": "🟠", "desc": "Condição potencialmente grave", "tempo": "Atendimento em até 10 min"},
    "Urgente": {"cor": "🟡", "desc": "Condição que pode evoluir", "tempo": "Atendimento em até 60 min"},
    "Pouco Urgente": {"cor": "🟢", "desc": "Condição estável", "tempo": "Atendimento em até 120 min"},
    "Não Urgente": {"cor": "🔵", "desc": "Condição benigna", "tempo": "Atendimento em até 240 min"}
}

# ───────────────────────────────────────────────────────────────
# FUNÇÕES UTILITÁRIAS
# ───────────────────────────────────────────────────────────────

def anonimizar_texto(texto):
    """Remove ou mascara informações de identificação pessoal (PHI)."""
    texto_anon = texto
    mascaras = {}

    for tipo, padrao in PHI_PATTERNS.items():
        matches = re.finditer(padrao, texto_anon, re.IGNORECASE)
        for i, match in enumerate(matches):
            chave = f"[{tipo.upper()}_{i+1}]"
            mascaras[chave] = match.group()
            texto_anon = texto_anon.replace(match.group(), chave, 1)

    return texto_anon, mascaras

def desanonimizar_texto(texto_anon, mascaras):
    """Restaura as informações originais (uso interno apenas)."""
    texto = texto_anon
    for chave, valor in mascaras.items():
        texto = texto.replace(chave, valor)
    return texto

def highlight_entidades(texto):
    """Destaca entidades médicas no texto com cores."""
    texto_highlight = texto
    cores = {
        "sintomas": "🟥",
        "diagnosticos": "🟦", 
        "medicamentos": "🟩",
        "procedimentos": "🟨",
        "exames_lab": "🟪"
    }

    for categoria, termos in ENTIDADES_MEDICAS.items():
        for termo in termos:
            # Case-insensitive replace com preservação do original
            pattern = re.compile(re.escape(termo), re.IGNORECASE)
            texto_highlight = pattern.sub(
                f"**{cores[categoria]} {termo.upper()}**", 
                texto_highlight
            )
    return texto_highlight

def extrair_entidades(texto):
    """Extrai e conta entidades médicas encontradas no texto."""
    encontradas = {cat: [] for cat in ENTIDADES_MEDICAS}
    texto_lower = texto.lower()

    for categoria, termos in ENTIDADES_MEDICAS.items():
        for termo in termos:
            if termo.lower() in texto_lower:
                encontradas[categoria].append(termo)

    return encontradas

def calcular_score_risco(texto):
    """Heurística simples para score de risco baseado em palavras-chave."""
    texto_lower = texto.lower()
    score = 0

    palavras_criticas = [
        "parada cardíaca", "parada cardiaca", "PCR", "ressuscitação",
        "choque", "hipotensão", "hipotensao", "taquicardia", "bradicardia",
        "arritmia", "fibrilação ventricular", "infarto", "AVC", "derrame",
        "hemorragia", "sangramento ativo", "perda de consciência", "inconsciente",
        "convulsão", "status epilepticus", "coma", "GCS", "glicose <", "hipoglicemia"
    ]

    palavras_urgentes = [
        "febre alta", "dispneia", "dor torácica", "dor toracica", "taquipneia",
        "desidratação", "desidratacao", "vômito persistente", "vomito persistente",
        "diarreia sanguinolenta", "icterícia", "ictericia", "edema agudo"
    ]

    for palavra in palavras_criticas:
        if palavra.lower() in texto_lower:
            score += 3
    for palavra in palavras_urgentes:
        if palavra.lower() in texto_lower:
            score += 2

    # Limita score máximo
    return min(score, 10)

def classificar_urgencia(score):
    """Classifica o nível de urgência baseado no score."""
    if score >= 6:
        return "Emergência"
    elif score >= 4:
        return "Muito Urgente"
    elif score >= 2:
        return "Urgente"
    elif score >= 1:
        return "Pouco Urgente"
    else:
        return "Não Urgente"

def gerar_resumo_clinico(texto):
    """Gera um resumo estruturado do registro clínico."""
    entidades = extrair_entidades(texto)
    score = calcular_score_risco(texto)
    urgencia = classificar_urgencia(score)

    resumo = {
        "urgencia": urgencia,
        "score_risco": score,
        "entidades": entidades,
        "contagem_entidades": {k: len(v) for k, v in entidades.items()}
    }
    return resumo

# ───────────────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analisador Semântico Clínico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("🔒 Privacidade")
    anonimizar = st.toggle("Anonimizar dados sensíveis (PHI)", value=True,
                           help="Remove CPF, telefone, email e nomes dos textos")

    st.subheader("🧠 Modelo de IA")
    modelo_llm = st.selectbox(
        "Modelo LLM:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )

    st.subheader("📊 Análises")
    mostrar_entidades = st.toggle("Destacar entidades médicas", value=True)
    mostrar_risco = st.toggle("Calcular score de risco", value=True)
    mostrar_clusters = st.toggle("Agrupar textos similares", value=True)

    st.markdown("---")
    st.info("""
    **💡 Dica:** Cole registros clínicos, evoluções, laudos ou anamneses.
    O sistema identifica automaticamente entidades médicas e calcula risco.
    """)

# Título principal
st.title("🏥 Analisador Semântico de Textos Clínicos com IA")
st.markdown("""
Ferramenta de **Processamento de Linguagem Natural (PLN)** especializada para a **área da saúde**.
Analise similaridade semântica entre registros clínicos, extraia entidades médicas, 
classifique risco e detecte contradições em laudos.
""")

# Área de entrada de textos com exemplo clínico
exemplo_default = (
    "Paciente masculino, 58 anos, relata dor torácica em aperto irradiada para o braço esquerdo há 30 min. "
    "PA: 180/110 mmHg, FC: 110 bpm, SatO2: 92%. ECG com supradesnivelamento de ST em V1-V4. "
    "Iniciado AAS 200mg + clopidogrel 300mg. Encaminhado para cateterismo de urgência.\n"
    "\n"
    "Paciente feminina, 34 anos, com queixa de cefaleia leve e náusea após stress no trabalho. "
    "PA: 120/80 mmHg, FC: 78 bpm. Sem alterações no exame físico. Prescrito paracetamol 750mg e repouso. "
    "Orientação de retorno em caso de piora.\n"
    "\n"
    "Paciente idoso, 72 anos, apresenta confusão mental aguda, febre de 39°C e tosse produtiva. "
    "PA: 90/60 mmHg (hipotenso), FC: 120 bpm, FR: 28 rpm. Raio-X com consolidação em lobo inferior direito. "
    "Diagnóstico: pneumonia comunitária grave. Iniciado ceftriaxone + azitromicina. Internação em UTI.\n"
    "\n"
    "Paciente masculino, 45 anos, consulta de rotina para acompanhamento de diabetes tipo 2. "
    "HBA1c: 7.2%. Em uso de metformina 850mg 2x/dia. Sem queixas atuais. "
    "Manteve medicação e agendou retorno em 3 meses."
)

texts = st.text_area(
    "📝 Cole os registros clínicos (um por linha ou parágrafo):",
    exemplo_default,
    height=250
)

# Botão de análise
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analisar = st.button("🔬 Analisar Registros Clínicos", type="primary", use_container_width=True)

if analisar:
    # Pré-processamento
    lines_raw = [t.strip() for t in texts.split("\n") if t.strip()]

    if len(lines_raw) < 1:
        st.warning("⚠️ Insira pelo menos 1 registro clínico.")
        st.stop()

    # Anonimização
    lines = []
    mascaras_totais = []

    for line in lines_raw:
        if anonimizar:
            line_anon, mascaras = anonimizar_texto(line)
            lines.append(line_anon)
            mascaras_totais.append(mascaras)
        else:
            lines.append(line)
            mascaras_totais.append({})

    # ───────────────────────────────────────────────────────────
    # ANÁLISE DE SIMILARIDADE SEMÂNTICA
    # ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Análise de Similaridade Semântica")

    embeddings = embed_model.encode(lines)
    sim_matrix = cosine_similarity(embeddings)

    col_matriz, col_info = st.columns([2, 1])

    with col_matriz:
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        sns.heatmap(
            sim_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            vmin=0, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            mask=mask
        )
        ax.set_xticks(range(len(lines)))
        ax.set_yticks(range(len(lines)))
        labels = [f"R{i+1}" for i in range(len(lines))]
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)
        ax.set_title("Matriz de Similaridade Cosseno (Registros Clínicos)", pad=20)
        st.pyplot(fig)

    with col_info:
        st.subheader("📈 Estatísticas")
        # Pares mais similares
        pares = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                pares.append((i, j, sim_matrix[i][j]))
        pares.sort(key=lambda x: x[2], reverse=True)

        st.markdown("**🔝 Pares mais similares:**")
        for i, j, score in pares[:3]:
            st.markdown(f"• R{i+1} ↔ R{j+1}: `{score:.3f}`")

        st.markdown("**🔻 Pares menos similares:**")
        for i, j, score in pares[-3:]:
            st.markdown(f"• R{i+1} ↔ R{j+1}: `{score:.3f}`")

        # Média de similaridade
        sim_media = np.mean([s for _, _, s in pares])
        st.metric("Similaridade Média", f"{sim_media:.3f}")

    # ───────────────────────────────────────────────────────────
    # CLUSTERING DE REGISTROS
    # ───────────────────────────────────────────────────────────
    if mostrar_clusters and len(lines) >= 3:
        st.subheader("🗂️ Agrupamento Semântico (Clusters)")

        n_clusters = min(3, len(lines) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        df_clusters = pd.DataFrame({
            "Registro": [f"R{i+1}" for i in range(len(lines))],
            "Cluster": clusters,
            "Resumo": [line[:80] + "..." if len(line) > 80 else line for line in lines]
        })

        for c in range(n_clusters):
            with st.expander(f"📦 Cluster {c+1}"):
                regs = df_clusters[df_clusters["Cluster"] == c]
                for _, row in regs.iterrows():
                    st.markdown(f"- **{row['Registro']}**: {row['Resumo']}")

    # ───────────────────────────────────────────────────────────
    # ANÁLISE INDIVIDUAL DE CADA REGISTRO
    # ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📋 Análise Individual dos Registros")

    for idx, (line, mascaras) in enumerate(zip(lines, mascaras_totais)):
        with st.expander(f"🏥 Registro R{idx+1}", expanded=(idx == 0)):
            col_texto, col_analise = st.columns([2, 1])

            with col_texto:
                st.markdown("**📝 Texto:**")
                if mostrar_entidades:
                    st.markdown(highlight_entidades(line))
                else:
                    st.markdown(f"```\n{line}\n```")

                if anonimizar and mascaras:
                    st.caption("🔒 Dados sensíveis anonimizados")

            with col_analise:
                # Score de risco
                if mostrar_risco:
                    score = calcular_score_risco(line)
                    urgencia = classificar_urgencia(score)
                    info_urg = ESCALA_URGENCIA[urgencia]

                    st.markdown(f"**{info_urg['cor']} Nível de Urgência: {urgencia}**")
                    st.progress(score / 10, text=f"Score de Risco: {score}/10")
                    st.caption(f"⏱️ {info_urg['tempo']}")

                # Entidades extraídas
                entidades = extrair_entidades(line)
                st.markdown("**🔍 Entidades Identificadas:**")
                for cat, vals in entidades.items():
                    if vals:
                        emoji = {"sintomas": "🟥", "diagnosticos": "🟦", 
                                "medicamentos": "🟩", "procedimentos": "🟨", 
                                "exames_lab": "🟪"}[cat]
                        st.markdown(f"{emoji} **{cat.capitalize()}:** {', '.join(vals)}")

    # ───────────────────────────────────────────────────────────
    # INTERPRETAÇÃO DA LLM
    # ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🤖 Interpretação Clínica da IA")

    # Construir prompt especializado para saúde
    prompt_saude = f"""Você é um especialista em Processamento de Linguagem Natural (PLN) aplicado à saúde, 
com experiência em análise de registros clínicos, evoluções médicas e laudos.

Analise semanticamente os seguintes registros clínicos e forneça:

1. **Similaridade e Agrupamento**: Quais registros tratam de condições/profiles clínicos similares? 
   Agrupe por afinidade semântica e justifique.

2. **Contradições e Inconsistências**: Há alguma contradição entre os registros? 
   (ex: diagnósticos conflitantes, medicações incompatíveis, evoluções contraditórias)

3. **Classificação por Severidade**: Ordene os registros do mais grave ao menos grave, 
   justificando com base nos sinais vitais, sintomas e diagnósticos mencionados.

4. **Entidades Médicas**: Liste os principais sintomas, diagnósticos, medicamentos e 
   procedimentos identificados em cada registro.

5. **Sugestão de Tags Semânticas**: Atribua 2-3 tags descritivas para cada registro 
   (ex: "#emergência_cardíaca", "#acompanhamento_rotina", "#pneumonia_grave").

6. **Alertas de Segurança**: Identifique potenciais erros de medicação, 
   interações medicamentosas ou sinais de alarme clínico que merecem atenção.

Registros Clínicos:
"""

    for i, line in enumerate(lines):
        prompt_saude += f"\n--- REGISTRO R{i+1} ---\n{line}\n"

    prompt_saude += """
\nFormate a resposta de forma clara e estruturada, usando markdown.
Seja preciso e baseie-se estritamente nas informações fornecidas nos registros.
"""

    with st.spinner("🧠 Consultando modelo de IA para análise clínica..."):
        try:
            resp = client.chat.completions.create(
                model=modelo_llm,
                messages=[
                    {"role": "system", "content": "Você é um especialista em PLN médico, análise clínica e semântica de registros de saúde. Responda em português do Brasil."},
                    {"role": "user", "content": prompt_saude}
                ],
                temperature=0.3,
                max_tokens=2500
            )

            resposta_llm = resp.choices[0].message.content

            # Tabs para organizar a resposta
            tab1, tab2 = st.tabs(["📄 Análise Completa", "📝 Prompt Enviado"])

            with tab1:
                st.markdown(resposta_llm)

            with tab2:
                st.code(prompt_saude, language="markdown")

        except Exception as e:
            st.error(f"❌ Erro ao consultar a IA: {e}")
            st.info("Verifique sua chave de API da OpenAI e tente novamente.")

    # ───────────────────────────────────────────────────────────
    # EXPORTAÇÃO DOS RESULTADOS
    # ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💾 Exportar Resultados")

    # Preparar dados para exportação
    dados_export = []
    for i, (line, mascaras) in enumerate(zip(lines, mascaras_totais)):
        entidades = extrair_entidades(line)
        score = calcular_score_risco(line)
        urgencia = classificar_urgencia(score)

        dados_export.append({
            "Registro": f"R{i+1}",
            "Texto": line,
            "Texto_Original": desanonimizar_texto(line, mascaras) if mascaras else line,
            "Score_Risco": score,
            "Urgencia": urgencia,
            "Sintomas": ", ".join(entidades["sintomas"]),
            "Diagnosticos": ", ".join(entidades["diagnosticos"]),
            "Medicamentos": ", ".join(entidades["medicamentos"]),
            "Procedimentos": ", ".join(entidades["procedimentos"]),
            "Exames_Lab": ", ".join(entidades["exames_lab"])
        })

    df_export = pd.DataFrame(dados_export)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv = df_export.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name="analise_clinica.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        json_str = df_export.to_json(orient="records", force_ascii=False, indent=2)
        st.download_button(
            label="📥 Baixar JSON",
            data=json_str,
            file_name="analise_clinica.json",
            mime="application/json",
            use_container_width=True
        )

    # Preview da tabela
    st.dataframe(df_export[["Registro", "Urgencia", "Score_Risco", "Sintomas", "Diagnosticos", "Medicamentos"]], 
                 use_container_width=True, hide_index=True)

# Rodapé
st.markdown("---")
st.caption("🏥 Analisador Semântico Clínico v2.0 | Desenvolvido para PLN em Saúde | Dados processados localmente, apenas embeddings e LLM usam API externa")
