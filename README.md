
---

```markdown
# 🏥 Analisador Semântico de Textos Clínicos com IA

> **Ferramenta de Processamento de Linguagem Natural (PLN)** especializada para a área da saúde. Analise similaridade semântica entre registros clínicos, extraia entidades médicas, classifique risco e detecte contradições em laudos — tudo com uma interface web intuitiva.

---

## 📑 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Demonstração](#-demonstração)
- [Tecnologias](#-tecnologias)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Arquitetura](#-arquitetura)
- [Dicionário de Entidades Médicas](#-dicionário-de-entidades-médicas)
- [Escala de Urgência](#-escala-de-urgência)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Contribuição](#-contribuição)
- [Licença](#-licença)
- [Agradecimentos](#-agradecimentos)

---

## 🔭 Visão Geral

Este projeto nasceu da necessidade de **automatizar a análise de registros clínicos** em português, permitindo que profissionais de saúde e pesquisadores identifiquem padrões, similaridades e riscos em grandes volumes de textos médicos de forma rápida e segura.

A ferramenta combina **embeddings semânticos**, **clustering de machine learning** e **Grandes Modelos de Linguagem (LLMs)** para oferecer insights clínicos acionáveis.

### 🎯 Público-alvo

- Médicos e enfermeiros
- Pesquisadores em saúde pública
- Desenvolvedores de sistemas hospitalares (HIS)
- Estudantes de medicina e computação
- Profissionais de qualidade e auditoria clínica

---

## ✨ Funcionalidades

| Funcionalidade | Descrição | Status |
|---|---|---|
| 🔒 **Anonimização de PHI** | Remove automaticamente CPF, telefone, email, datas e nomes de pacientes | ✅ |
| 🟥🟦🟩 **Extração de Entidades Médicas** | Identifica e destaca sintomas, diagnósticos, medicamentos, procedimentos e exames laboratoriais | ✅ |
| 📊 **Matriz de Similaridade** | Calcula e visualiza o grau de semelhança semântica entre todos os registros | ✅ |
| 🗂️ **Clustering Automático** | Agrupa registros clínicos similares usando K-Means | ✅ |
| 🔴🟠🟡 **Score de Risco & Triagem** | Classifica urgência com base em palavras-chave críticas (Emergência → Não Urgente) | ✅ |
| 🤖 **Interpretação por IA** | GPT-4o-mini analisa contradições, severidade e sugere tags semânticas | ✅ |
| 💾 **Exportação de Dados** | Download dos resultados em CSV e JSON para integração com outros sistemas | ✅ |
| ⚙️ **Configurações Personalizáveis** | Sidebar com toggles para anonimização, destaque de entidades, cálculo de risco e modelo LLM | ✅ |

---

## 🎬 Demonstração

```
┌─────────────────────────────────────────────────────────────┐
│  🏥 Analisador Semântico de Textos Clínicos com IA          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [📝 Área de texto com registros clínicos]                  │
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │  🔬 Analisar Registros Clínicos         │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  📊 Matriz de Similaridade (Heatmap RdYlGn)                │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │     │ R1  │ R2  │ R3  │                                 │
│  ├─────┼─────┼─────┼─────┤                                 │
│  │ R1  │ 1.0 │0.85 │0.32 │                                 │
│  │ R2  │     │ 1.0 │0.28 │                                 │
│  │ R3  │     │     │ 1.0 │                                 │
│  └─────┴─────┴─────┴─────┘                                 │
│                                                             │
│  🗂️ Cluster 1: R1, R2 (Emergências cardíacas)              │
│  🗂️ Cluster 2: R3 (Acompanhamento de rotina)               │
│                                                             │
│  🔴 R1 — Emergência (Score: 9/10)                          │
│  🟡 R2 — Urgente (Score: 5/10)                             │
│  🔵 R3 — Não Urgente (Score: 1/10)                         │
│                                                             │
│  🤖 [Análise completa da IA com 6 seções]                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tecnologias

### Core

| Tecnologia | Versão | Propósito |
|---|---|---|
| [Python](https://www.python.org/) | 3.9+ | Linguagem principal |
| [Streamlit](https://streamlit.io/) | 1.28+ | Interface web interativa |
| [Sentence-Transformers](https://www.sbert.net/) | 2.2+ | Geração de embeddings semânticos |
| [OpenAI](https://openai.com/) | 1.0+ | Interpretação clínica via GPT-4o-mini |

### Análise de Dados

| Tecnologia | Propósito |
|---|---|
| [NumPy](https://numpy.org/) | Computação numérica e manipulação de embeddings |
| [Pandas](https://pandas.pydata.org/) | Estruturação e exportação de resultados |
| [Scikit-learn](https://scikit-learn.org/) | Cálculo de similaridade cosseno e clustering K-Means |

### Visualização

| Tecnologia | Propósito |
|---|---|
| [Matplotlib](https://matplotlib.org/) | Gráficos base (matriz de calor) |
| [Seaborn](https://seaborn.pydata.org/) | Heatmap estatístico profissional |

### Utilitários

| Tecnologia | Propósito |
|---|---|
| [python-dotenv](https://saurabh-kumar.com/python-dotenv/) | Gerenciamento seguro de variáveis de ambiente |
| [re](https://docs.python.org/3/library/re.html) (built-in) | Expressões regulares para anonimização de PHI |

---

## 📋 Pré-requisitos

- **Python** 3.9 ou superior
- **Chave de API da OpenAI** ([obtenha aqui](https://platform.openai.com/api-keys))
- **Git** (opcional, para clonar o repositório)
- **Navegador web** moderno (Chrome, Firefox, Edge)

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/analisador-semantico-saude.git
cd analisador-semantico-saude
```

### 2. Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 **Dica:** Se o PowerShell bloquear a ativação, use o CMD ou execute:
> ```powershell
> powershell -ExecutionPolicy Bypass -Command ".\venv\Scripts\activate"
> ```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn sentence-transformers openai python-dotenv
```

> ⏱️ **Nota:** A primeira execução do `sentence-transformers` fará o download automático do modelo `all-MiniLM-L6-v2` (~80 MB).

---

## 🔐 Configuração

Crie um arquivo `.env` na raiz do projeto com sua chave da OpenAI:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

> ⚠️ **Nunca** commite o arquivo `.env`! Ele já está no `.gitignore`.

### Modelo de embeddings alternativo (opcional)

Para resultados ainda melhores em português clínico, você pode usar um modelo biomedical:

```env
EMBED_MODEL=neuml/pubmedbert-base-embeddings
OPENAI_API_KEY=sk-sua-chave-aqui
```

---

## ▶️ Como Usar

### Iniciar a aplicação

```bash
streamlit run semantic_analyzer_health.py
```

Ou, se o comando `streamlit` não estiver no PATH:

```bash
python -m streamlit run semantic_analyzer_health.py
```

### Acessar no navegador

O Streamlit abrirá automaticamente em:

```
🌐 Local URL: http://localhost:8501
🌐 Network URL: http://192.168.x.x:8501
```

### Fluxo de uso

1. **Cole os registros clínicos** na área de texto (um por linha/parágrafo)
2. **Ajuste as configurações** na sidebar (anonimização, modelo LLM, etc.)
3. **Clique em "🔬 Analisar Registros Clínicos"**
4. **Explore os resultados:**
   - Matriz de similaridade (heatmap)
   - Clusters semânticos
   - Score de risco por registro
   - Entidades médicas destacadas
   - Interpretação completa da IA
5. **Exporte os dados** em CSV ou JSON

---

## 📁 Estrutura do Projeto

```
analisador-semantico-saude/
│
├── 📄 semantic_analyzer_health.py   # Código principal da aplicação
├── 📄 requirements.txt              # Dependências do projeto
├── 📄 .env.example                  # Template de variáveis de ambiente
├── 📄 .gitignore                    # Arquivos ignorados pelo Git
├── 📄 README.md                     # Este arquivo
│
├── 📁 assets/                       # Imagens e screenshots
│   ├── screenshot_heatmap.png
│   ├── screenshot_analise.png
│   └── demo.gif
│
└── 📁 docs/                         # Documentação adicional
    ├── arquitetura.md
    └── guia_contribuicao.md
```

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT (Frontend)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Text Area   │  │ Sidebar     │  │ Buttons     │  │ Charts    │ │
│  │ (Input)     │  │ (Config)    │  │ (Actions)   │  │ (Output)  │ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└─────────┼───────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE PROCESSAMENTO                      │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │ Anonimização│───▶│ Embeddings  │───▶│ Similaridade Cosseno    │ │
│  │ (re + PHI)  │    │ (SBERT)     │    │ (sklearn)               │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                               │                     │
│                                               ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │ Extração de │◄───│ Clustering  │◄───│ Matriz de Similaridade  │ │
│  │ Entidades   │    │ (K-Means)   │    │                         │ │
│  │ (Dicionário)│    └─────────────┘    └─────────────────────────┘ │
│  └─────────────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐    ┌─────────────────────────────────────────────┐ │
│  │ Score de    │───▶│ Interpretação Clínica (OpenAI GPT-4o-mini)  │ │
│  │ Risco       │    │ • Similaridade e agrupamento                  │ │
│  │ (Heurística)│    │ • Contradições e inconsistências              │ │
│  └─────────────┘    │ • Classificação por severidade                │ │
│                     │ • Alertas de segurança                        │ │
│                     └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPORTAÇÃO                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │ CSV         │  │ JSON        │  │ DataFrame (Preview)         │ │
│  │ (Pandas)    │  │ (json)      │  │ (Tabela interativa)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Dicionário de Entidades Médicas

O sistema reconhece automaticamente **150+ termos** organizados em 5 categorias:

### 🟥 Sintomas
Febra, dor torácica, dispneia, cefaleia, náusea, vômito, diarreia, constipação, tosse, cansaço, fadiga, tontura, palpitação, edema, erupção, prurido, calafrio, sudorese, perda de peso, inchaço, hematoma, sangramento, convulsão, desmaio, confusão, agitação, depressão, ansiedade, insônia, anorexia, poliúria, polidipsia, polifagia, dor abdominal, dor lombar, dor articular, mialgia, artralgia...

### 🟦 Diagnósticos
Hipertensão, diabetes, asma, bronquite, pneumonia, infarto, AVC, derrame, câncer, tumor, neoplasia, anemia, leucemia, hepatite, cirrose, nefrite, insuficiência renal, insuficiência cardíaca, arritmia, fibrilação, estenose, valvopatia, osteoporose, artrite, gota, lúpus, esclerose, esquizofrenia, bipolar, demência, Alzheimer, Parkinson, epilepsia, migrânea...

### 🟩 Medicamentos
Paracetamol, dipirona, ibuprofeno, aspirina, amoxicilina, azitromicina, ciprofloxacino, metformina, insulina, losartana, enalapril, atenolol, amlodipino, simvastatina, atorvastatina, omeprazol, prednisona, hidrocortisona, dexametasona, warfarina, rivaroxabana, heparina, clopidogrel, fluoxetina, sertralina, risperidona, haloperidol, diazepam, lorazepam, tramadol, morfina, fentanil, ondansetrona, metoclopramida...

### 🟨 Procedimentos
Eletrocardiograma, ecocardiograma, raio-X, tomografia, ressonância, ultrassom, endoscopia, colonoscopia, biópsia, cirurgia, cateterismo, diálise, quimioterapia, radioterapia, fisioterapia, hemodiálise, ventilação mecânica, intubação, traqueostomia, punção, drenagem, sutura...

### 🟪 Exames Laboratoriais
Hemograma, glicemia, colesterol, triglicerídeos, creatinina, ureia, TGO, TGP, bilirrubina, TSH, T4, T3, PSA, CEA, CA-125, HbA1c, PCR, VHS, INR, PTT, plaquetas, leucócitos, neutrófilos, linfócitos, eosinófilos, basófilos, eritrócitos, hemoglobina, hematócrito, VCM, HCM, RDW...

---

## 🚨 Escala de Urgência

| Nível | Cor | Descrição | Tempo de Atendimento |
|---|---|---|---|
| **Emergência** | 🔴 | Risco imediato de vida | Atendimento imediato |
| **Muito Urgente** | 🟠 | Condição potencialmente grave | Até 10 minutos |
| **Urgente** | 🟡 | Condição que pode evoluir | Até 60 minutos |
| **Pouco Urgente** | 🟢 | Condição estável | Até 120 minutos |
| **Não Urgente** | 🔵 | Condição benigna | Até 240 minutos |

> O score de risco é calculado por heurística baseada em palavras-chave críticas encontradas no texto (parada cardíaca, choque, hemorragia, coma, etc.).

---

## 💡 Exemplos de Uso

### Exemplo 1: Triagem em Pronto-Socorro

```text
Paciente masculino, 58 anos, relata dor torácica em aperto irradiada 
para o braço esquerdo há 30 min. PA: 180/110 mmHg, FC: 110 bpm, 
SatO2: 92%. ECG com supradesnivelamento de ST em V1-V4. Iniciado 
AAS 200mg + clopidogrel 300mg. Encaminhado para cateterismo de urgência.
```

**Resultado:** 🔴 **Emergência** | Score: 9/10  
**Entidades:** dor torácica, supradesnivelamento de ST, AAS, clopidogrel, cateterismo  
**Cluster:** Emergências cardíacas

---

### Exemplo 2: Acompanhamento de Rotina

```text
Paciente masculino, 45 anos, consulta de rotina para acompanhamento 
de diabetes tipo 2. HbA1c: 7.2%. Em uso de metformina 850mg 2x/dia. 
Sem queixas atuais. Manteve medicação e agendou retorno em 3 meses.
```

**Resultado:** 🔵 **Não Urgente** | Score: 1/10  
**Entidades:** diabetes tipo 2, HbA1c, metformina  
**Cluster:** Acompanhamento crônico

---

### Exemplo 3: Detecção de Contradição

```text
Registro 1: "Paciente com diagnóstico de diabetes tipo 1, em uso de 
insulina NPH 20UI manhã e 16UI noite."

Registro 2 (mesmo paciente, 2 dias depois): "Paciente com diabetes 
tipo 2 controlada apenas com metformina 850mg, sem uso de insulina."
```

**Alerta da IA:** ⚠️ **Contradição detectada** — Tipo de diabetes e esquema medicamentoso inconsistentes entre registros.

---



## 🤝 Contribuição

Contribuições são bem-vindas! Siga os passos:

1. **Fork** o projeto
2. Crie uma **branch** (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um **Pull Request**

### Ideias para contribuir

- [ ] Adicionar mais entidades médicas ao dicionário
- [ ] Suporte a múltiplos idiomas (espanhol, inglês)
- [ ] Integração com FHIR (padrão de interoperabilidade em saúde)
- [ ] Modelo de embeddings em português clínico (BioBERT-pt)
- [ ] Dashboard com histórico de análises
- [ ] Modo batch para processar arquivos CSV/Excel

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** — veja o arquivo [LICENSE](LICENSE) para detalhes.

> ⚠️ **Aviso legal:** Esta ferramenta é destinada a fins educacionais e de pesquisa. **Não substitui o julgamento clínico de um profissional de saúde.** Sempre valide os resultados antes de tomar decisões médicas.

---

## 🙏 Agradecimentos

- [Sentence-Transformers](https://www.sbert.net/) por embeddings de alta qualidade
- [OpenAI](https://openai.com/) pelos modelos GPT
- [Streamlit](https://streamlit.io/) pela simplicidade na criação de interfaces
- Comunidade open-source de PLN e saúde

---

<p align="center">
  Feito com ❤️ e 🧠 para a área da saúde
  <br>
  <a href="https://github.com/seu-usuario/analisador-semantico-saude">⭐ Star este projeto</a>
</p>
```

---
# analisador-semantico-saude
