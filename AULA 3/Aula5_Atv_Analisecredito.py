"""
app.py - Simulador de crédito para financiamento da primeira moto.

O código está dividido em quatro camadas, sempre nesta ordem:
    1. Regras de negócio  -> funções puras, sem Streamlit (fáceis de testar)
    2. Validação          -> confere se os dados digitados fazem sentido
    3. Interface          -> formulário que coleta os dados do usuário
    4. Exibição           -> mostra o resultado (mensagens, métricas e gráfico)

Para executar:  streamlit run app.py
"""

# Permite usar a sintaxe "float | None" nas anotações de tipo em versões mais antigas do Python
from __future__ import annotations

from dataclasses import dataclass  # cria classes simples que só guardam dados
from enum import Enum  # cria conjuntos fixos de valores (os possíveis status)

import pandas as pd  # já vem instalado junto com o Streamlit; usado no gráfico de barras
import streamlit as st  # biblioteca que transforma o script em aplicação web


# =============================================================================
# 1. REGRAS DE NEGÓCIO
# =============================================================================

# Constantes ficam no topo: se a regra mudar, alteramos em um único lugar
IDADE_MINIMA: int = 18  # maioridade civil exigida para assinar o contrato
PERCENTUAL_MAXIMO_RENDA: float = 0.30  # a parcela pode comprometer no máximo 30% da renda


class Status(Enum):
    """Os três desfechos possíveis da análise."""

    APROVADO = "aprovado"  # passou nas duas regras
    NEGADO_IDADE = "negado_idade"  # reprovado por ser menor de 18 anos
    NEGADO_RENDA = "negado_renda"  # reprovado por comprometer mais de 30% da renda


@dataclass(frozen=True)  # frozen=True impede que o resultado seja alterado depois de criado
class ResultadoAnalise:
    """Guarda tudo o que a camada de exibição precisa saber sobre a análise."""

    status: Status  # qual foi o desfecho da análise
    parcela_maxima: float  # renda * 30%
    comprometimento: float  # fração da renda que a parcela consome (0.25 = 25%)
    parcela_sugerida: float | None = None  # preenchida só na negação por renda
    anos_para_maioridade: int | None = None  # preenchida só na negação por idade

    @property
    def aprovado(self) -> bool:
        """Atalho para saber se o crédito foi aprovado."""
        return self.status is Status.APROVADO


def calcular_parcela_maxima(renda: float) -> float:
    """Regra 2: parcela máxima permitida = renda * 30%."""
    # round(..., 2) evita ruídos de ponto flutuante (ex.: 300.03000000000003)
    return round(renda * PERCENTUAL_MAXIMO_RENDA, 2)


def calcular_comprometimento(parcela: float, renda: float) -> float:
    """Calcula que fração da renda mensal a parcela consome."""
    if renda <= 0:  # evita divisão por zero e resultados sem sentido
        raise ValueError("A renda mensal deve ser maior que zero.")
    return parcela / renda  # ex.: parcela 600 / renda 3000 = 0.20 (20%)


def analisar_credito(idade: int, renda: float, parcela: float) -> ResultadoAnalise:
    """Aplica as regras de negócio na ordem correta e devolve o resultado."""
    parcela = round(parcela, 2)  # mesma precisão usada na parcela máxima, para comparar igual com igual
    parcela_maxima = calcular_parcela_maxima(renda)  # limite de parcela para essa renda
    comprometimento = calcular_comprometimento(parcela, renda)  # % da renda que seria comprometida

    # Regra 1: menor de idade é negado automaticamente, sem olhar a renda
    if idade < IDADE_MINIMA:
        return ResultadoAnalise(
            status=Status.NEGADO_IDADE,
            parcela_maxima=parcela_maxima,
            comprometimento=comprometimento,
            anos_para_maioridade=IDADE_MINIMA - idade,  # quantos anos faltam para os 18
        )

    # Regra 2 + 3: maior de idade, mas a parcela passa do limite -> nega e sugere o valor viável
    if parcela > parcela_maxima:
        return ResultadoAnalise(
            status=Status.NEGADO_RENDA,
            parcela_maxima=parcela_maxima,
            comprometimento=comprometimento,
            parcela_sugerida=parcela_maxima,  # a sugestão é a própria parcela máxima
        )

    # Regra 4: idade >= 18 e parcela <= limite -> aprovado
    return ResultadoAnalise(
        status=Status.APROVADO,
        parcela_maxima=parcela_maxima,
        comprometimento=comprometimento,
    )


# =============================================================================
# 2. VALIDAÇÃO DOS DADOS
# =============================================================================

def validar_entradas(renda: float, parcela: float) -> list[str]:
    """Devolve uma lista de mensagens de erro (lista vazia = dados válidos)."""
    erros: list[str] = []  # começa sem erros e vamos acrescentando
    if renda <= 0:  # sem renda não há como calcular o comprometimento
        erros.append("Informe uma renda mensal maior que zero.")
    if parcela <= 0:  # parcela zerada não representa um financiamento
        erros.append("Informe um valor de parcela maior que zero.")
    return erros


# =============================================================================
# 3. INTERFACE (COLETA DE DADOS)
# =============================================================================

def configurar_pagina() -> None:
    """Configura título da aba e largura da página. Deve ser a primeira chamada ao Streamlit."""
    st.set_page_config(page_title="Simulador de financiamento de moto", page_icon="🏍️", layout="centered")


def renderizar_cabecalho() -> None:
    """Desenha o título e a explicação inicial."""
    st.title("🏍️ Simulador de crédito para moto")  # título principal da página
    st.caption(  # texto pequeno e discreto abaixo do título
        "Informe seus dados e veja se o financiamento seria aprovado. "
        "Esta é uma simulação educativa e não representa uma análise real de banco."
    )


def coletar_dados() -> tuple[bool, int, float, float]:
    """Mostra o formulário e devolve (enviado, idade, renda, parcela)."""
    # st.form agrupa os campos: a página só recalcula quando o botão é clicado
    with st.form("form_simulacao"):
        idade = st.number_input(
            "Idade (anos)",
            min_value=0,  # não aceita idade negativa
            max_value=120,  # limite razoável para evitar digitação errada
            value=20,  # valor inicial exibido
            step=1,  # passo dos botões + e -
        )
        renda = st.number_input(
            "Renda mensal (R$)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=3000.0,
            step=100.0,
            format="%.2f",  # sempre exibe duas casas decimais
        )
        parcela = st.number_input(
            "Valor da parcela desejada (R$)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=800.0,
            step=50.0,
            format="%.2f",
        )
        enviado = st.form_submit_button("Simular financiamento")  # True só no clique

    # int() garante idade inteira; float() garante números decimais para as contas
    return enviado, int(idade), float(renda), float(parcela)


# =============================================================================
# 4. EXIBIÇÃO DO RESULTADO
# =============================================================================

def formatar_moeda(valor: float) -> str:
    """Formata no padrão brasileiro: 1234.5 -> 'R$ 1.234,50'."""
    texto = f"{valor:,.2f}"  # formato americano: 1,234.50
    # Troca vírgula e ponto usando "X" como marcador temporário para um não sobrescrever o outro
    return "R$ " + texto.replace(",", "X").replace(".", ",").replace("X", ".")


def exibir_mensagem_principal(resultado: ResultadoAnalise, renda: float, parcela: float) -> None:
    """Mostra o banner de aprovado/negado e, se houver, a sugestão de ajuste."""
    if resultado.status is Status.APROVADO:
        st.success(  # caixa verde
            f"✅ **Crédito aprovado!** A parcela de {formatar_moeda(parcela)} compromete "
            f"{resultado.comprometimento:.1%} da sua renda, dentro do limite de "
            f"{PERCENTUAL_MAXIMO_RENDA:.0%}."
        )

    elif resultado.status is Status.NEGADO_IDADE:
        st.error(  # caixa vermelha
            "❌ **Crédito negado.** É necessário ter 18 anos ou mais (maioridade civil) "
            "para contratar um financiamento."
        )
        anos = resultado.anos_para_maioridade or 0  # "or 0" evita None na conta abaixo
        unidade = "ano" if anos == 1 else "anos"  # concordância: 1 ano / 2 anos
        st.info(  # caixa azul com a sugestão
            f"💡 **Sugestão:** refaça a simulação ao completar 18 anos (faltam {anos} {unidade})."
        )

    else:  # Status.NEGADO_RENDA
        st.error(
            f"❌ **Crédito negado.** A parcela de {formatar_moeda(parcela)} compromete "
            f"{resultado.comprometimento:.1%} da sua renda, acima do limite de "
            f"{PERCENTUAL_MAXIMO_RENDA:.0%}."
        )
        st.info(
            f"💡 **Sugestão:** com renda de {formatar_moeda(renda)}, a parcela máxima viável "
            f"é de **{formatar_moeda(resultado.parcela_sugerida or 0.0)}**. "
            "Refaça a simulação com esse valor ou menor."
        )


def exibir_indicadores(resultado: ResultadoAnalise, parcela: float) -> None:
    """Mostra métricas, a barra de uso do limite e o gráfico comparativo."""
    col_desejada, col_maxima, col_percentual = st.columns(3)  # três colunas lado a lado
    col_desejada.metric("Parcela desejada", formatar_moeda(parcela))
    col_maxima.metric("Parcela máxima permitida", formatar_moeda(resultado.parcela_maxima))
    col_percentual.metric("Renda comprometida", f"{resultado.comprometimento:.1%}")

    # A barra representa o uso do limite de 30%: 100% cheia = exatamente no limite
    uso_do_limite = resultado.comprometimento / PERCENTUAL_MAXIMO_RENDA
    st.progress(
        min(uso_do_limite, 1.0),  # progress só aceita valores entre 0.0 e 1.0
        text=f"Uso do limite de comprometimento de renda: {uso_do_limite:.0%}",
    )

    # Gráfico de barras nativo do Streamlit (sem matplotlib): parcela desejada x limite
    dados = pd.DataFrame(
        {"Valor (R$)": [parcela, resultado.parcela_maxima]},
        index=["Parcela desejada", "Parcela máxima permitida"],
    )
    st.bar_chart(dados)


def exibir_resultado(resultado: ResultadoAnalise, renda: float, parcela: float) -> None:
    """Orquestra tudo o que aparece depois da análise."""
    st.divider()  # linha horizontal separando o formulário do resultado
    st.subheader("Resultado da análise")
    exibir_mensagem_principal(resultado, renda, parcela)

    # Na negação por idade a renda nem chega a ser avaliada, então não mostramos os indicadores
    if resultado.status is not Status.NEGADO_IDADE:
        exibir_indicadores(resultado, parcela)


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

def main() -> None:
    """Fluxo principal: interface -> validação -> regras de negócio -> exibição."""
    configurar_pagina()
    renderizar_cabecalho()

    enviado, idade, renda, parcela = coletar_dados()  # desenha o formulário e lê os valores

    if not enviado:  # antes do clique, só pedimos o preenchimento
        st.info("Preencha os campos e clique em **Simular financiamento**.")
        return

    erros = validar_entradas(renda, parcela)  # confere renda e parcela
    if erros:  # se houver problemas, explicamos o que corrigir e paramos aqui
        for erro in erros:
            st.error(erro)
        return

    resultado = analisar_credito(idade, renda, parcela)  # aplica as regras de negócio
    exibir_resultado(resultado, renda, parcela)  # mostra o resultado na tela


# O Streamlit executa o arquivo como script principal, então este bloco roda normalmente.
# Ele também permite importar as funções de regra em testes sem abrir a interface.
if __name__ == "__main__":
    main()