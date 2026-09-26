"""
atividade5_deteccao_spacy.py - Detector de Reclamações com spaCy.

ATIVIDADE 5 - demonstra spaCy + tokenização + normalização + condição +
detecção de palavras negativas + Streamlit, sem Machine Learning e sem
análise de sentimento avançada.

O programa não interpreta o sentido da frase: ele só verifica, token a
token, se alguma palavra já normalizada (minúscula) está no conjunto
PALAVRAS_NEGATIVAS, definido diretamente no código (não é um dicionário
léxico externo). É comparação de texto com uma estrutura condicional
if / else — nada além disso.

Não usa tradução automática, banco de dados, nem qualquer API externa.

Fluxo da aplicação:

    Mensagem
        |
    Processamento com spaCy (Doc / Token)
        |
    Normalização (minúsculas) e filtro (somente palavras, sem pontuação)
        |
    Comparação com PALAVRAS_NEGATIVAS
        |
    Regra condicional (if / else)
        |
    Resultado

Para executar:  streamlit run atividade5_deteccao_spacy.py
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from __future__ import annotations

import html as html_lib  # escapa texto do usuário antes de inserir em HTML

import spacy
import streamlit as st


# =============================================================================
# 2. CONFIGURAÇÃO DA APLICAÇÃO
# =============================================================================

LIMITE_CARACTERES = 5000  # limite de segurança para mensagens muito longas
NOME_MODELO_SPACY = "pt_core_news_sm"  # modelo de português usado pelo spaCy

CSS_PERSONALIZADO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.stApp {
    background-color: #F3F5F7;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.cabecalho-titulo {
    font-size: 2.1rem;
    font-weight: 700;
    color: #16212E;
    margin: 0.25rem 0 0.1rem 0;
    letter-spacing: -0.01em;
}
.cabecalho-titulo::after {
    content: "";
    display: block;
    width: 46px;
    height: 4px;
    background-color: #0F766E;
    margin-top: 0.55rem;
    border-radius: 2px;
}
.cabecalho-subtitulo {
    font-size: 1.02rem;
    color: #56636F;
    max-width: 620px;
    margin: 0.9rem 0 1.25rem 0;
    line-height: 1.55;
}

.secao-titulo {
    font-size: 1rem;
    font-weight: 600;
    color: #16212E;
    margin: 0 0 0.7rem 0;
}

.bloco-texto {
    background-color: #F7F9FA;
    border: 1px solid #E2E8ED;
    border-left: 4px solid #94A3B0;
    border-radius: 6px;
    padding: 0.75rem 0.95rem;
    font-size: 0.95rem;
    color: #24313D;
    line-height: 1.6;
    white-space: pre-wrap;
}
.bloco-tokens { border-left-color: #64748B; }
.bloco-negativas { border-left-color: #B42318; }

.stButton > button {
    background-color: #0F766E;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #0B5C55;
    color: #FFFFFF;
}
</style>
"""


def configurar_pagina() -> None:
    """Define título da aba e largura da página. Deve ser a primeira chamada ao Streamlit."""
    st.set_page_config(page_title="Detector de Reclamações", layout="centered")


def aplicar_estilo() -> None:
    """Injeta o CSS personalizado da aplicação (fundo, cabeçalho, blocos e botão)."""
    st.markdown(CSS_PERSONALIZADO, unsafe_allow_html=True)


def renderizar_cabecalho() -> None:
    """Desenha o título principal e o subtítulo explicativo da ferramenta."""
    st.markdown(
        '<p class="cabecalho-titulo">Detector de Reclamações</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="cabecalho-subtitulo">Análise de palavras negativas com spaCy.</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. CARREGAMENTO DO MODELO SPACY
# =============================================================================

@st.cache_resource(show_spinner="Carregando o modelo de português do spaCy...")
def carregar_pipeline_spacy():
    """Carrega o modelo de português do spaCy.

    O decorador @st.cache_resource garante que o modelo seja carregado uma
    única vez por sessão, e não a cada clique no botão.
    """
    return spacy.load(NOME_MODELO_SPACY)


# =============================================================================
# 4. LISTA DE PALAVRAS NEGATIVAS
# =============================================================================

# Estrutura simples de dados definida diretamente no código (não é um dicionário
# léxico externo). Um set é usado porque a verificação "a palavra está aqui?"
# (operador `in`) é feita para cada token da mensagem, e essa verificação é
# mais rápida em um conjunto do que em uma lista.
PALAVRAS_NEGATIVAS = {
    "ruim",
    "erro",
    "péssimo",
}


# =============================================================================
# 5. PROCESSAMENTO COM SPACY (TOKENIZAÇÃO)
# =============================================================================

def processar_com_spacy(nlp, texto: str):
    """Processa o texto com o pipeline do spaCy e devolve o objeto Doc.

    O Doc é a representação do texto já tokenizado: iterar sobre ele devolve
    um Token para cada unidade identificada (palavras, números, pontuação).
    """
    return nlp(texto)


def obter_tokens_como_texto(doc) -> list[str]:
    """Devolve o texto de cada Token do Doc, na ordem original, sem nenhum filtro.

    Isso mostra exatamente o que o spaCy identificou como token — incluindo
    pontuação — para o estudante visualizar o funcionamento da tokenização.
    """
    return [token.text for token in doc]


# =============================================================================
# 6. NORMALIZAÇÃO E FILTRO
# =============================================================================

def normalizar_e_filtrar_palavras(doc) -> list[str]:
    """Prepara os tokens para a comparação com PALAVRAS_NEGATIVAS.

    Duas transformações são feitas aqui, e só aqui (a lista de tokens exibida
    ao usuário continua intacta):
    - token.is_alpha mantém apenas tokens que são palavras, descartando
      pontuação e números, para que um "." nunca seja comparado ao léxico;
    - token.text.lower() normaliza a palavra para minúsculas, para que
      "RUIM", "Ruim" e "ruim" sejam tratados como a mesma palavra.
    """
    return [token.text.lower() for token in doc if token.is_alpha]


# =============================================================================
# 7. DETECÇÃO DE PALAVRAS NEGATIVAS (REGRA CONDICIONAL)
# =============================================================================

def detectar_palavras_negativas(palavras_normalizadas: list[str]) -> list[str]:
    """Percorre as palavras normalizadas e verifica cada uma contra PALAVRAS_NEGATIVAS.

    Usa a estrutura condicional pedida pela atividade: para cada palavra, ou
    ela está no conjunto de palavras negativas (e é adicionada à lista de
    resultados), ou não está (e é simplesmente ignorada). Repetições são
    mantidas: se "ruim" aparecer duas vezes na mensagem, ela aparece duas
    vezes na lista devolvida.
    """
    palavras_negativas_encontradas: list[str] = []
    for palavra in palavras_normalizadas:
        if palavra in PALAVRAS_NEGATIVAS:
            palavras_negativas_encontradas.append(palavra)
        else:
            continue  # palavra não está na lista de negativas: segue para a próxima
    return palavras_negativas_encontradas


def obter_palavras_distintas(palavras_negativas_encontradas: list[str]) -> list[str]:
    """Remove duplicatas da lista de ocorrências, preservando a ordem de aparição.

    Serve só para exibição: deixa claro, separadamente, quais palavras
    diferentes foram encontradas, sem se confundir com a quantidade total de
    ocorrências (que pode ser maior, se alguma palavra se repetir).
    """
    return list(dict.fromkeys(palavras_negativas_encontradas))


def mensagem_e_reclamacao(palavras_negativas_encontradas: list[str]) -> bool:
    """Decide se a mensagem deve ser sinalizada como possível reclamação."""
    if palavras_negativas_encontradas:
        e_reclamacao = True
    else:
        e_reclamacao = False
    return e_reclamacao


# =============================================================================
# 8. VALIDAÇÃO DE ENTRADA
# =============================================================================

def validar_entrada(texto: str) -> list[str]:
    """Devolve uma lista de mensagens de erro (lista vazia = entrada válida)."""
    erros: list[str] = []
    if not texto.strip():
        erros.append("Digite ou cole uma mensagem para realizar a análise.")
    elif len(texto) > LIMITE_CARACTERES:
        erros.append(
            f"A mensagem tem {len(texto)} caracteres; o limite é {LIMITE_CARACTERES}. "
            "Divida-a em partes menores."
        )
    return erros


# =============================================================================
# 9. INTERFACE STREAMLIT
# =============================================================================

def coletar_entrada() -> tuple[bool, str]:
    """Mostra o formulário e devolve (enviado, texto_digitado)."""
    with st.form("form_deteccao"):
        texto_original = st.text_area(
            "Digite a mensagem do cliente:",
            height=150,
            placeholder="Ex.: O produto apresentou um erro e está ruim.",
        )
        enviado = st.form_submit_button("Analisar mensagem")
    return enviado, texto_original


# =============================================================================
# 10. EXIBIÇÃO DOS RESULTADOS
# =============================================================================

def bloco_texto(texto: str, classe_extra: str) -> None:
    """Mostra um texto dentro de um bloco estilizado, escapando HTML por segurança."""
    texto_seguro = html_lib.escape(texto)
    st.markdown(f'<div class="bloco-texto {classe_extra}">{texto_seguro}</div>', unsafe_allow_html=True)


def exibir_tokens(tokens: list[str]) -> None:
    """Mostra, dentro de um cartão, os tokens identificados pelo spaCy."""
    with st.container(border=True):
        st.markdown('<p class="secao-titulo">Tokens identificados</p>', unsafe_allow_html=True)
        bloco_texto(" | ".join(tokens), "bloco-tokens")
        st.caption(f"{len(tokens)} tokens no total, incluindo pontuação.")


def exibir_resultado(
    palavras_negativas_encontradas: list[str], palavras_distintas: list[str], e_reclamacao: bool
) -> None:
    """Mostra, dentro de um cartão, o status da mensagem e as palavras negativas."""
    with st.container(border=True):
        st.markdown('<p class="secao-titulo">Resultado</p>', unsafe_allow_html=True)

        col_ocorrencias, col_distintas = st.columns(2)
        col_ocorrencias.metric("Ocorrências de palavras negativas", len(palavras_negativas_encontradas))
        col_distintas.metric("Palavras negativas distintas", len(palavras_distintas))

        if e_reclamacao:
            st.warning("Esta mensagem foi identificada como uma possível reclamação.")
            bloco_texto(", ".join(palavras_negativas_encontradas), "bloco-negativas")
            st.caption("Palavras distintas (sem repetição): " + ", ".join(palavras_distintas))
        else:
            st.success("Nenhuma palavra negativa foi encontrada nesta mensagem.")

        st.caption(
            "Este resultado vem de uma comparação de palavras, não de uma análise de "
            "sentimento: o spaCy não interpreta o significado da frase, apenas identifica "
            "os tokens que coincidem com PALAVRAS_NEGATIVAS."
        )


# =============================================================================
# 11. PROCESSAMENTO DA MENSAGEM + PONTO DE ENTRADA
# =============================================================================

def main() -> None:
    """Fluxo principal: interface -> validação -> spaCy -> normalização/filtro ->
    detecção -> exibição."""
    configurar_pagina()
    aplicar_estilo()
    renderizar_cabecalho()

    try:
        nlp = carregar_pipeline_spacy()
    except OSError:
        st.error(
            f'O modelo "{NOME_MODELO_SPACY}" do spaCy não foi encontrado. Instale-o '
            f"executando no terminal: python -m spacy download {NOME_MODELO_SPACY} — "
            "depois recarregue esta página."
        )
        st.stop()

    enviado, texto_original = coletar_entrada()

    if not enviado:
        st.info('Digite uma mensagem e clique em "Analisar mensagem".')
        return

    erros = validar_entrada(texto_original)
    if erros:
        for erro in erros:
            st.error(erro)
        return

    try:
        doc = processar_com_spacy(nlp, texto_original)
    except Exception:
        st.error("Ocorreu um problema inesperado ao analisar a mensagem. Tente novamente em instantes.")
        return

    tokens = obter_tokens_como_texto(doc)
    palavras_normalizadas = normalizar_e_filtrar_palavras(doc)
    palavras_negativas_encontradas = detectar_palavras_negativas(palavras_normalizadas)
    palavras_distintas = obter_palavras_distintas(palavras_negativas_encontradas)
    e_reclamacao = mensagem_e_reclamacao(palavras_negativas_encontradas)

    st.divider()
    exibir_resultado(palavras_negativas_encontradas, palavras_distintas, e_reclamacao)
    exibir_tokens(tokens)


# O Streamlit executa o arquivo como script principal, então este bloco roda normalmente.
# Ele também permite importar as funções de lógica em testes sem abrir a interface.
if __name__ == "__main__":
    main()