
#Atividade 2

"""
tokenizador_app.py - Tokenizador de mensagens de clientes com tradução automática.

Fluxo da aplicação:
    1. O analista cola a mensagem do cliente (em português ou inglês).
    2. A mensagem é traduzida para português com deep-translator, garantindo
       que todas as mensagens sejam analisadas em um único idioma.
    3. O texto traduzido é tokenizado (separado em palavras) com o NLTK.
    4. A aplicação exibe os tokens e as palavras-chave mais frequentes.

Para executar:  streamlit run tokenizador_app.py
"""

from __future__ import annotations

import string  # fornece string.punctuation, usada para filtrar sinais de pontuação
from collections import Counter  # conta quantas vezes cada palavra aparece

import pandas as pd  # usado para montar a tabela de tokens e o gráfico de frequência
import streamlit as st  # biblioteca que transforma o script em aplicação web

import nltk
from nltk.corpus import stopwords  # lista de palavras muito comuns ("de", "a", "o"...)
from nltk.tokenize import word_tokenize  # função que separa o texto em tokens

from deep_translator import GoogleTranslator  # tradutor automático baseado no Google Tradutor


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

IDIOMA_DESTINO = "pt"  # todas as mensagens são padronizadas para português
IDIOMA_NLTK = "portuguese"  # nome do idioma no formato esperado pelo NLTK
LIMITE_CARACTERES = 5000  # limite de segurança para a API de tradução gratuita
TOP_N_PALAVRAS_CHAVE = 10  # quantas palavras-chave exibir no gráfico

# Pacotes do NLTK necessários e o caminho usado para verificar se já estão instalados.
# "punkt_tab" só existe em versões mais recentes do NLTK; por isso é tratado como opcional.
RECURSOS_NLTK = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]


# =============================================================================
# PREPARAÇÃO DO NLTK
# =============================================================================

@st.cache_resource(show_spinner="Preparando recursos de linguagem (primeira execução)...")
def preparar_recursos_nltk() -> None:
    """Garante que os pacotes do NLTK estejam baixados. Roda apenas uma vez por sessão.

    O decorador @st.cache_resource evita baixar os pacotes novamente a cada
    interação do usuário com a interface.
    """
    for caminho_dados, nome_pacote in RECURSOS_NLTK:
        try:
            nltk.data.find(caminho_dados)  # já instalado: não faz nada
        except LookupError:
            try:
                nltk.download(nome_pacote, quiet=True)  # baixa o pacote silenciosamente
            except Exception:
                # "punkt_tab" é opcional (não existe em versões antigas do NLTK).
                # Os demais pacotes são obrigatórios: se falharem, a exceção sobe
                # e é tratada em main(), avisando o usuário.
                if nome_pacote != "punkt_tab":
                    raise


# =============================================================================
# TRADUÇÃO
# =============================================================================

def traduzir_para_portugues(texto: str) -> str:
    """Traduz o texto para português, detectando o idioma de origem automaticamente.

    Se o texto já estiver em português, o tradutor normalmente o devolve
    praticamente inalterado — o que também serve para padronizar pequenas
    variações de escrita.
    """
    tradutor = GoogleTranslator(source="auto", target=IDIOMA_DESTINO)
    return tradutor.translate(texto)


# =============================================================================
# TOKENIZAÇÃO E PALAVRAS-CHAVE
# =============================================================================

def tokenizar_texto(texto: str, remover_pontuacao: bool = True) -> list[str]:
    """Separa o texto em tokens (palavras e sinais de pontuação) usando o NLTK."""
    tokens = word_tokenize(texto, language=IDIOMA_NLTK)
    if remover_pontuacao:
        # Mantém apenas os tokens que não são puramente pontuação (ex.: remove "," e ".",
        # mas mantém "não" e "2x1")
        tokens = [token for token in tokens if token not in string.punctuation]
    return tokens


def extrair_palavras_chave(tokens: list[str], top_n: int = TOP_N_PALAVRAS_CHAVE) -> list[tuple[str, int]]:
    """Filtra pontuação, números e palavras muito comuns para destacar os termos relevantes.

    Esta lista é usada apenas no gráfico de "palavras-chave" — a lista completa
    de tokens (exigida pela tarefa) continua sendo exibida sem esse filtro.
    """
    stopwords_pt = set(stopwords.words(IDIOMA_NLTK))
    palavras_relevantes = [
        token.lower()
        for token in tokens
        if token.isalpha()  # descarta números e pontuação
        and token.lower() not in stopwords_pt  # descarta palavras muito comuns
    ]
    contagem = Counter(palavras_relevantes)
    return contagem.most_common(top_n)


# =============================================================================
# VALIDAÇÃO
# =============================================================================

def validar_entrada(texto: str) -> list[str]:
    """Devolve uma lista de mensagens de erro (lista vazia = entrada válida)."""
    erros: list[str] = []
    if not texto.strip():
        erros.append("Digite ou cole uma mensagem antes de processar.")
    elif len(texto) > LIMITE_CARACTERES:
        erros.append(
            f"A mensagem tem {len(texto)} caracteres; o limite é {LIMITE_CARACTERES}. "
            "Divida-a em partes menores."
        )
    return erros


# =============================================================================
# INTERFACE (COLETA DE DADOS)
# =============================================================================

def configurar_pagina() -> None:
    """Configura título da aba e largura da página. Deve ser a primeira chamada ao Streamlit."""
    st.set_page_config(page_title="Tokenizador de mensagens", page_icon="✂️", layout="centered")


def renderizar_cabecalho() -> None:
    """Desenha o título e a explicação inicial da ferramenta."""
    st.title("✂️ Tokenizador de mensagens de clientes")
    st.caption(
        "Cole a mensagem do cliente (em português ou inglês). A ferramenta traduz o texto "
        "para português e separa o conteúdo em tokens (palavras) para facilitar a análise."
    )


def coletar_entrada() -> tuple[bool, str, bool]:
    """Mostra o formulário e devolve (enviado, texto_digitado, remover_pontuacao)."""
    with st.form("form_tokenizacao"):
        texto = st.text_area(
            "Mensagem do cliente",
            height=160,
            placeholder="Ex.: My motorcycle stopped working after only two days of use.",
        )
        remover_pontuacao = st.checkbox(
            "Remover pontuação da lista de tokens", value=True
        )
        enviado = st.form_submit_button("Processar mensagem")
    return enviado, texto, remover_pontuacao


# =============================================================================
# EXIBIÇÃO DO RESULTADO
# =============================================================================

def exibir_texto_traduzido(texto_original: str, texto_traduzido: str) -> None:
    """Mostra o texto já padronizado em português e, opcionalmente, o original."""
    st.subheader("Texto padronizado (traduzido)")
    st.write(texto_traduzido)

    # Só vale a pena mostrar o texto original se ele for diferente do traduzido
    if texto_original.strip() != texto_traduzido.strip():
        with st.expander("Ver mensagem original enviada pelo cliente"):
            st.write(texto_original)


def exibir_tokens(tokens: list[str]) -> None:
    """Mostra as métricas gerais e a tabela completa de tokens extraídos."""
    col_total, col_unicos = st.columns(2)
    col_total.metric("Total de tokens", len(tokens))
    col_unicos.metric("Palavras únicas", len({token.lower() for token in tokens}))

    st.subheader(f"Tokens extraídos ({len(tokens)})")
    tabela_tokens = pd.DataFrame({"#": range(1, len(tokens) + 1), "Token": tokens})
    st.dataframe(tabela_tokens, hide_index=True, use_container_width=True)


def exibir_palavras_chave(palavras_chave: list[tuple[str, int]]) -> None:
    """Mostra um gráfico de barras com as palavras-chave mais frequentes."""
    if not palavras_chave:  # acontece se a mensagem só tiver pontuação/stopwords
        st.info("Nenhuma palavra-chave relevante foi encontrada nesta mensagem.")
        return

    st.subheader("Palavras-chave mais frequentes")
    st.caption("Pontuação e palavras muito comuns (\"de\", \"para\", \"o\"...) foram removidas.")
    tabela_frequencia = pd.DataFrame(palavras_chave, columns=["Palavra", "Frequência"])
    tabela_frequencia = tabela_frequencia.set_index("Palavra")
    st.bar_chart(tabela_frequencia)


def exibir_resultado(
    texto_original: str,
    texto_traduzido: str,
    tokens: list[str],
    palavras_chave: list[tuple[str, int]],
) -> None:
    """Orquestra tudo o que aparece depois do processamento da mensagem."""
    st.divider()
    exibir_texto_traduzido(texto_original, texto_traduzido)
    exibir_tokens(tokens)
    exibir_palavras_chave(palavras_chave)


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

def main() -> None:
    """Fluxo principal: interface -> validação -> tradução -> tokenização -> exibição."""
    configurar_pagina()
    renderizar_cabecalho()

    # Baixa os pacotes do NLTK antes de mostrar o formulário (só acontece de fato na 1ª vez)
    try:
        preparar_recursos_nltk()
    except Exception:
        st.error(
            "Não foi possível preparar os recursos de linguagem do NLTK. "
            "Verifique a conexão com a internet e recarregue a página."
        )
        st.stop()  # interrompe a execução: sem o NLTK a tokenização não funciona

    enviado, texto, remover_pontuacao = coletar_entrada()

    if not enviado:  # antes do primeiro clique, só orientamos o preenchimento
        st.info("Cole a mensagem do cliente e clique em **Processar mensagem**.")
        return

    erros = validar_entrada(texto)
    if erros:
        for erro in erros:
            st.error(erro)
        return

    # Tenta traduzir; se a API de tradução falhar (ex.: sem internet), avisa o
    # analista e continua a análise com o texto original, em vez de travar o app.
    with st.spinner("Traduzindo mensagem..."):
        try:
            texto_traduzido = traduzir_para_portugues(texto)
        except Exception:
            st.warning(
                "Não foi possível traduzir a mensagem agora (verifique a conexão com a "
                "internet). Exibindo a tokenização do texto original, sem tradução."
            )
            texto_traduzido = texto

    tokens = tokenizar_texto(texto_traduzido, remover_pontuacao)
    palavras_chave = extrair_palavras_chave(tokens)

    exibir_resultado(texto, texto_traduzido, tokens, palavras_chave)


# O Streamlit executa o arquivo como script principal, então este bloco roda normalmente.
# Ele também permite importar as funções de lógica em testes sem abrir a interface.
if __name__ == "__main__":
    main()

#Atividade 3

