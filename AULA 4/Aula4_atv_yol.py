"""
atividade_deteccao_objetos_simulada.py - Simulador de Detecção de Objetos.

ATIVIDADE - simula, de forma simplificada e baseada em REGRAS (sem machine
learning e sem deep learning), como um sistema poderia tentar identificar se
uma imagem contém determinados objetos (pessoa, carro, animal).

Isto NÃO é um detector de objetos real. Detectores reais (YOLO, SSD,
Faster R-CNN, etc.) usam redes neurais treinadas com milhões de imagens
rotuladas. Aqui usamos só estatísticas simples da imagem -- cor média,
proporção (largura x altura) e densidade de bordas -- comparadas com
limiares fixos definidos manualmente no código. O objetivo é didático:
mostrar como um sistema de "detecção" pode ser estruturado (entrada ->
extração de características -> regras -> saída) e, principalmente, por
que essa abordagem simples é limitada e não substitui um modelo treinado.

AVISO IMPORTANTE: este simulador não deve ser usado para decisões reais,
segurança, vigilância ou qualquer aplicação que dependa de precisão. Os
resultados são apenas ilustrativos.

Fluxo da aplicação:

    Imagem enviada pelo usuário
            |
    Extração de características (cor média, proporção, densidade de bordas)
            |
    Comparação com regras fixas por categoria (pessoa / carro / animal)
            |
    Resultado (quais regras "bateram" e por quê)

Para executar:  streamlit run atividade_deteccao_objetos_simulada.py
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from PIL import Image, ImageFilter, ImageStat


# =============================================================================
# 2. CONFIGURAÇÃO DA APLICAÇÃO
# =============================================================================

LIMITE_TAMANHO_MB = 8  # limite de segurança para o upload

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
    max-width: 640px;
    margin: 0.9rem 0 1.1rem 0;
    line-height: 1.55;
}

.secao-titulo {
    font-size: 1rem;
    font-weight: 600;
    color: #16212E;
    margin: 0 0 0.7rem 0;
}

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
    st.set_page_config(page_title="Simulador de Detecção de Objetos", layout="centered")


def aplicar_estilo() -> None:
    """Injeta o CSS personalizado da aplicação."""
    st.markdown(CSS_PERSONALIZADO, unsafe_allow_html=True)


def renderizar_cabecalho() -> None:
    """Desenha o título, o subtítulo e o aviso sobre a natureza simulada do sistema."""
    st.markdown(
        '<p class="cabecalho-titulo">Simulador de Detecção de Objetos</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="cabecalho-subtitulo">Simulação de visão computacional baseada em regras '
        "fixas sobre estatísticas da imagem, sem machine learning ou deep learning.</p>",
        unsafe_allow_html=True,
    )
    st.warning(
        "Este é um sistema SIMULADO para fins didáticos. Ele não reconhece o conteúdo real "
        "da imagem -- apenas compara cor média, proporção e densidade de bordas com limiares "
        "fixos. Não use para decisões reais, segurança ou qualquer aplicação que dependa de "
        "precisão."
    )


# =============================================================================
# 3. EXTRAÇÃO DE CARACTERÍSTICAS DA IMAGEM
# =============================================================================

@dataclass
class EstatisticasImagem:
    """Características simples extraídas da imagem, usadas pelas regras a seguir."""

    largura: int
    altura: int
    cor_media: tuple[int, int, int]  # média de R, G, B
    variedade_cor: float  # média do desvio padrão dos 3 canais de cor
    densidade_bordas: float  # 0.0 (imagem lisa) a 1.0 (imagem muito texturizada)


def carregar_imagem(arquivo_enviado) -> Image.Image:
    """Abre o arquivo enviado como uma imagem Pillow em modo RGB."""
    imagem = Image.open(arquivo_enviado)
    return imagem.convert("RGB")


def calcular_estatisticas(imagem: Image.Image) -> EstatisticasImagem:
    """Calcula as características simples usadas pelas regras de detecção.

    A imagem é reduzida antes dos cálculos só para acelerar o processamento
    de fotos grandes -- isso não muda a essência das estatísticas (cor média
    e textura aproximada continuam representativas da imagem original).
    """
    largura, altura = imagem.size
    imagem_reduzida = imagem.copy()
    imagem_reduzida.thumbnail((200, 200))

    estatisticas_cor = ImageStat.Stat(imagem_reduzida)
    cor_media = tuple(int(valor) for valor in estatisticas_cor.mean)  # (R, G, B)
    variedade_cor = sum(estatisticas_cor.stddev) / len(estatisticas_cor.stddev)

    # ImageFilter.FIND_EDGES realça contornos; quanto mais clara (em média) a
    # imagem resultante, mais bordas/textura a imagem original tinha.
    imagem_cinza = imagem_reduzida.convert("L")
    imagem_bordas = imagem_cinza.filter(ImageFilter.FIND_EDGES)
    densidade_bordas = ImageStat.Stat(imagem_bordas).mean[0] / 255.0

    return EstatisticasImagem(
        largura=largura,
        altura=altura,
        cor_media=cor_media,
        variedade_cor=variedade_cor,
        densidade_bordas=densidade_bordas,
    )


def distancia_cor(cor_a: tuple[int, int, int], cor_b: tuple[int, int, int]) -> float:
    """Distância euclidiana entre duas cores RGB -- quanto menor, mais parecidas."""
    return sum((a - b) ** 2 for a, b in zip(cor_a, cor_b)) ** 0.5


def cor_mais_proxima(cor_media: tuple[int, int, int], paleta: list[tuple[int, int, int]]) -> float:
    """Devolve a menor distância entre cor_media e qualquer cor de uma paleta de referência."""
    return min(distancia_cor(cor_media, cor_referencia) for cor_referencia in paleta)


# =============================================================================
# 4. REGRAS DE DETECÇÃO POR CATEGORIA
# =============================================================================

# Paletas de referência (aproximadas e ilustrativas -- não calibradas com dados reais).
CORES_REFERENCIA_CARRO = [
    (178, 34, 34),   # vermelho
    (25, 42, 86),    # azul
    (192, 192, 192), # prata
    (20, 20, 20),    # preto
    (240, 240, 240), # branco
]
CORES_REFERENCIA_ANIMAL = [
    (139, 90, 43),   # marrom
    (160, 120, 80),  # caramelo
    (105, 80, 55),   # marrom escuro
    (180, 150, 110), # bege
]

# Limiares usados pelas regras. São valores de exemplo, calibrados apenas com
# imagens sintéticas simples (cores sólidas, listras, blocos de ruído) -- não
# foram calibrados contra um conjunto real de fotos rotuladas. A escala de
# "densidade_bordas" é naturalmente baixa (o filtro FIND_EDGES do Pillow, em
# imagens comuns, raramente ultrapassa ~0.3), por isso os limiares abaixo são
# pequenos -- ajuste-os aqui caso teste com fotos reais.
LIMIAR_DISTANCIA_COR = 45.0
LIMIAR_PROPORCAO = 1.05
LIMIAR_BORDAS_BAIXA = 0.05
LIMIAR_BORDAS_ALTA = 0.07
LIMIAR_VARIEDADE_COR_PESSOA = 35.0
LIMIAR_BORDAS_PESSOA_MIN = 0.015
LIMIAR_BORDAS_PESSOA_MAX = 0.25


@dataclass
class ResultadoRegra:
    """Resultado da avaliação de uma regra: se ela bateu e por quê (para transparência)."""

    objeto: str
    correspondeu: bool
    motivo: str


def avaliar_regra_carro(estatisticas: EstatisticasImagem) -> ResultadoRegra:
    """Regra para 'carro': imagem em formato paisagem, cor próxima de uma cor
    típica de carro, e poucas bordas (superfícies lisas e uniformes)."""
    eh_paisagem = estatisticas.largura > estatisticas.altura * LIMIAR_PROPORCAO
    distancia = cor_mais_proxima(estatisticas.cor_media, CORES_REFERENCIA_CARRO)
    cor_compativel = distancia < LIMIAR_DISTANCIA_COR
    bordas_baixas = estatisticas.densidade_bordas < LIMIAR_BORDAS_BAIXA

    correspondeu = eh_paisagem and cor_compativel and bordas_baixas
    motivo = (
        f"proporção paisagem: {'sim' if eh_paisagem else 'não'} "
        f"({estatisticas.largura}x{estatisticas.altura}); "
        f"cor média {estatisticas.cor_media} a {distancia:.1f} de distância de uma cor "
        f"típica de carro (limite: < {LIMIAR_DISTANCIA_COR:.0f}); "
        f"densidade de bordas {estatisticas.densidade_bordas:.2f} "
        f"(limite: < {LIMIAR_BORDAS_BAIXA:.2f})"
    )
    return ResultadoRegra(objeto="carro", correspondeu=correspondeu, motivo=motivo)


def avaliar_regra_animal(estatisticas: EstatisticasImagem) -> ResultadoRegra:
    """Regra para 'animal': cor média em tons terrosos e muitas bordas
    (textura irregular, como pelagem)."""
    distancia = cor_mais_proxima(estatisticas.cor_media, CORES_REFERENCIA_ANIMAL)
    cor_compativel = distancia < LIMIAR_DISTANCIA_COR
    bordas_altas = estatisticas.densidade_bordas > LIMIAR_BORDAS_ALTA

    correspondeu = cor_compativel and bordas_altas
    motivo = (
        f"cor média {estatisticas.cor_media} a {distancia:.1f} de distância de um tom "
        f"terroso de referência (limite: < {LIMIAR_DISTANCIA_COR:.0f}); "
        f"densidade de bordas {estatisticas.densidade_bordas:.2f} "
        f"(limite: > {LIMIAR_BORDAS_ALTA:.2f})"
    )
    return ResultadoRegra(objeto="animal", correspondeu=correspondeu, motivo=motivo)


def avaliar_regra_pessoa(estatisticas: EstatisticasImagem) -> ResultadoRegra:
    """Regra para 'pessoa': imagem em formato retrato, muitas cores diferentes
    (roupas/fundo variados) e densidade de bordas moderada."""
    eh_retrato = estatisticas.altura > estatisticas.largura * LIMIAR_PROPORCAO
    cor_variada = estatisticas.variedade_cor > LIMIAR_VARIEDADE_COR_PESSOA
    bordas_na_faixa = LIMIAR_BORDAS_PESSOA_MIN <= estatisticas.densidade_bordas <= LIMIAR_BORDAS_PESSOA_MAX

    correspondeu = eh_retrato and cor_variada and bordas_na_faixa
    motivo = (
        f"proporção retrato: {'sim' if eh_retrato else 'não'} "
        f"({estatisticas.largura}x{estatisticas.altura}); "
        f"variedade de cor {estatisticas.variedade_cor:.1f} "
        f"(limite: > {LIMIAR_VARIEDADE_COR_PESSOA:.0f}); "
        f"densidade de bordas {estatisticas.densidade_bordas:.2f} "
        f"(faixa esperada: {LIMIAR_BORDAS_PESSOA_MIN:.2f} a {LIMIAR_BORDAS_PESSOA_MAX:.2f})"
    )
    return ResultadoRegra(objeto="pessoa", correspondeu=correspondeu, motivo=motivo)


def detectar_objetos_simulado(estatisticas: EstatisticasImagem) -> list[ResultadoRegra]:
    """Aplica todas as regras cadastradas e devolve o resultado de cada uma --
    tenha ela correspondido ou não. Mostrar as duas situações (e não só as que
    bateram) é o que torna a simulação transparente: o usuário vê exatamente
    por que cada objeto foi ou não sinalizado.
    """
    return [
        avaliar_regra_carro(estatisticas),
        avaliar_regra_animal(estatisticas),
        avaliar_regra_pessoa(estatisticas),
    ]


# =============================================================================
# TRATAMENTO DE ERROS - VALIDAÇÃO DE ENTRADA
# =============================================================================

def validar_arquivo(arquivo_enviado) -> list[str]:
    """Devolve uma lista de mensagens de erro (lista vazia = entrada válida)."""
    erros: list[str] = []
    if arquivo_enviado is None:
        erros.append("Envie uma imagem antes de analisar.")
        return erros

    tamanho_mb = arquivo_enviado.size / (1024 * 1024)
    if tamanho_mb > LIMITE_TAMANHO_MB:
        erros.append(f"A imagem tem {tamanho_mb:.1f} MB; o limite é {LIMITE_TAMANHO_MB} MB.")
    return erros


# =============================================================================
# 5. INTERFACE STREAMLIT
# =============================================================================

def coletar_entrada():
    """Mostra o formulário e devolve (enviado, arquivo_enviado)."""
    with st.form("form_deteccao"):
        arquivo_enviado = st.file_uploader("Envie uma imagem (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        enviado = st.form_submit_button("Analisar imagem")
    return enviado, arquivo_enviado


# =============================================================================
# 6. EXIBIÇÃO DOS RESULTADOS
# =============================================================================

def exibir_estatisticas(estatisticas: EstatisticasImagem) -> None:
    """Mostra, dentro de um cartão, as características extraídas da imagem."""
    with st.container(border=True):
        st.markdown(
            '<p class="secao-titulo">Características extraídas da imagem</p>',
            unsafe_allow_html=True,
        )
        col_dimensoes, col_cor, col_variedade, col_bordas = st.columns(4)
        col_dimensoes.metric("Dimensões", f"{estatisticas.largura}x{estatisticas.altura}")
        col_cor.metric("Cor média (RGB)", str(estatisticas.cor_media))
        col_variedade.metric("Variedade de cor", f"{estatisticas.variedade_cor:.1f}")
        col_bordas.metric("Densidade de bordas", f"{estatisticas.densidade_bordas:.2f}")


def exibir_resultados_deteccao(resultados: list[ResultadoRegra]) -> None:
    """Mostra, dentro de um cartão, o status geral e o detalhe de cada regra avaliada."""
    with st.container(border=True):
        st.markdown('<p class="secao-titulo">Resultado da simulação</p>', unsafe_allow_html=True)

        encontrados = [resultado for resultado in resultados if resultado.correspondeu]
        if encontrados:
            nomes = ", ".join(resultado.objeto for resultado in encontrados)
            st.warning(f"Objetos possivelmente identificados pelas regras: {nomes}")
        else:
            st.info("Nenhum objeto foi identificado pelas regras atuais.")

        for resultado in resultados:
            rotulo = "regra correspondeu" if resultado.correspondeu else "regra não correspondeu"
            with st.expander(f"{resultado.objeto} -- {rotulo}"):
                st.write(resultado.motivo)

        st.caption(
            "Isto é uma simulação baseada em regras fixas sobre estatísticas simples da "
            "imagem -- não é reconhecimento real de objetos. Um sistema real de visão "
            "computacional usa modelos treinados (redes neurais) em vez de limiares "
            "definidos manualmente."
        )


# =============================================================================
# 7. PROCESSAMENTO DA ENTRADA + PONTO DE ENTRADA
# =============================================================================

def main() -> None:
    """Fluxo principal: interface -> validação -> extração de características ->
    regras -> exibição."""
    configurar_pagina()
    aplicar_estilo()
    renderizar_cabecalho()

    enviado, arquivo_enviado = coletar_entrada()

    if not enviado:
        st.info('Envie uma imagem e clique em "Analisar imagem".')
        return

    erros = validar_arquivo(arquivo_enviado)
    if erros:
        for erro in erros:
            st.error(erro)
        return

    try:
        imagem = carregar_imagem(arquivo_enviado)
    except Exception:
        st.error("Não foi possível abrir esta imagem. Verifique se o arquivo não está corrompido.")
        return

    estatisticas = calcular_estatisticas(imagem)
    resultados = detectar_objetos_simulado(estatisticas)

    st.divider()
    st.image(imagem, caption="Imagem enviada", use_container_width=True)
    exibir_estatisticas(estatisticas)
    exibir_resultados_deteccao(resultados)


# O Streamlit executa o arquivo como script principal, então este bloco roda normalmente.
# Ele também permite importar as funções de lógica em testes sem abrir a interface.
if __name__ == "__main__":
    main()