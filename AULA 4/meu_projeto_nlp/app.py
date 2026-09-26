import sqlite3
import html
import os
import tkinter as tk
from tkinter import ttk, messagebox
import spacy

# ==============================================================================
# 1. NLP E REGRAS DE CLASSIFICAÇÃO (NLP)
# ==============================================================================
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

CATEGORIAS = {
    "Bloqueio de cartão": {
        "bloquear", "bloqueio", "cartao", "cartão", "roubo",
        "roubado", "furtado", "furto", "perdido", "perda"
    },
    "Segunda via de boleto": {
        "boleto", "segunda", "via", "emitir", "gerar", "novo"
    }
}

def classificar_mensagem(texto: str) -> dict:
    if not texto or not texto.strip():
        return {
            "categoria": "Entrada inválida",
            "palavras_encontradas": [],
            "frequencia": 0,
            "detalhes": "A mensagem fornecida está vazia."
        }

    doc = nlp(texto)
    tokens_normalizados = [
        token.text.lower() for token in doc 
        if not token.is_punct and not token.is_space
    ]

    correspondencias = {}
    todas_palavras = set()

    for categoria, palavras_chave in CATEGORIAS.items():
        palavras_casadas = [token for token in tokens_normalizados if token in palavras_chave]
        correspondencias[categoria] = palavras_casadas
        todas_palavras.update(palavras_casadas)

    pontuacoes = {cat: len(palavras) for cat, palavras in correspondencias.items()}
    max_pontuacao = max(pontuacoes.values()) if pontuacoes else 0

    if max_pontuacao == 0:
        return {
            "categoria": "Categoria não identificada",
            "palavras_encontradas": [],
            "frequencia": 0,
            "detalhes": "Nenhuma palavra-chave conhecida foi encontrada na mensagem."
        }

    categorias_vencedoras = [
        cat for cat, pontuacao in pontuacoes.items() if pontuacao == max_pontuacao
    ]

    if len(categorias_vencedoras) > 1:
        palavras_ambiguas = []
        for cat in categorias_vencedoras:
            palavras_ambiguas.extend(correspondencias[cat])
        
        return {
            "categoria": "Solicitação ambígua",
            "palavras_encontradas": sorted(list(set(palavras_ambiguas))),
            "frequencia": max_pontuacao,
            "detalhes": f"Pontuação equivalente para: {', '.join(categorias_vencedoras)}."
        }

    cat_final = categorias_vencedoras[0]
    palavras_finais = sorted(list(set(correspondencias[cat_final])))

    return {
        "categoria": cat_final,
        "palavras_encontradas": palavras_finais,
        "frequencia": len(palavras_finais),
        "detalhes": "Classificação baseada em regras de palavras-chave."
    }

# ==============================================================================
# 2. BANCO DE DADOS (MODEL)
# ==============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), "banco.db")

class SolicitacaoModel:
    @staticmethod
    def get_connection():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def inicializar_banco():
        query = """
        CREATE TABLE IF NOT EXISTS solicitacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mensagem TEXT NOT NULL,
            categoria TEXT NOT NULL,
            palavras_encontradas TEXT,
            data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with SolicitacaoModel.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()

    @staticmethod
    def salvar(mensagem: str, categoria: str, palavras_encontradas: str):
        query = """
        INSERT INTO solicitacoes (mensagem, categoria, palavras_encontradas)
        VALUES (?, ?, ?);
        """
        with SolicitacaoModel.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (mensagem, categoria, palavras_encontradas))
            conn.commit()

    @staticmethod
    def listar_todos():
        query = "SELECT id, mensagem, categoria, palavras_encontradas, data_hora FROM solicitacoes ORDER BY id DESC;"
        with SolicitacaoModel.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()

# ==============================================================================
# 3. INTERFACE GRÁFICA / VISUALIZAÇÃO (VIEW & CONTROLLER)
# ==============================================================================
class AppClassificador(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classificador de Solicitações Bancárias")
        self.geometry("800x650")
        self.configure(bg="#f4f6f9")

        SolicitacaoModel.inicializar_banco()
        self._criar_componentes()
        self._carregar_historico()

    def _criar_componentes(self):
        # Cabeçalho usando tk.Frame (espaçamento interno feito com padx/pady no pack)
        frame_header = tk.Frame(self, bg="#0d47a1")
        frame_header.pack(fill="x", padx=0, pady=0)
        
        lbl_titulo = tk.Label(
            frame_header, 
            text="Classificador de Solicitações Bancárias", 
            font=("Helvetica", 16, "bold"), 
            fg="white", 
            bg="#0d47a1"
        )
        lbl_titulo.pack(pady=(15, 2))

        lbl_sub = tk.Label(
            frame_header, 
            text="Processamento de Linguagem Natural com spaCy e SQLite", 
            font=("Helvetica", 10), 
            fg="#e3f2fd", 
            bg="#0d47a1"
        )
        lbl_sub.pack(pady=(0, 15))

        # Formúlário de Entrada
        frame_form = tk.LabelFrame(self, text=" Digitar Solicitação ", font=("Helvetica", 11, "bold"), bg="#f4f6f9", padx=10, pady=10)
        frame_form.pack(fill="x", padx=15, pady=10)

        self.txt_entrada = tk.Text(frame_form, height=4, font=("Helvetica", 10))
        self.txt_entrada.pack(fill="x", pady=5)

        btn_classificar = tk.Button(
            frame_form, 
            text="Classificar Solicitação", 
            font=("Helvetica", 10, "bold"), 
            bg="#0d47a1", 
            fg="white", 
            activebackground="#1565c0",
            activeforeground="white",
            command=self._processar_classificacao
        )
        btn_classificar.pack(fill="x", pady=5)

        # Painel de Resultado
        self.frame_res = tk.LabelFrame(self, text=" Resultado da Análise ", font=("Helvetica", 11, "bold"), bg="#f4f6f9", padx=10, pady=10)
        self.frame_res.pack(fill="x", padx=15, pady=5)

        self.lbl_categoria = tk.Label(self.frame_res, text="Categoria: -", font=("Helvetica", 10, "bold"), bg="#f4f6f9", anchor="w")
        self.lbl_categoria.pack(fill="x")

        self.lbl_palavras = tk.Label(self.frame_res, text="Palavras-chave: -", font=("Helvetica", 10), bg="#f4f6f9", anchor="w")
        self.lbl_palavras.pack(fill="x")

        # Tabela de Histórico
        frame_hist = tk.LabelFrame(self, text=" Histórico de Solicitações ", font=("Helvetica", 11, "bold"), bg="#f4f6f9", padx=10, pady=10)
        frame_hist.pack(fill="both", expand=True, padx=15, pady=10)

        colunas = ("id", "mensagem", "categoria", "palavras", "data")
        self.tabela = ttk.Treeview(frame_hist, columns=colunas, show="headings", height=8)

        self.tabela.heading("id", text="ID")
        self.tabela.heading("mensagem", text="Mensagem")
        self.tabela.heading("categoria", text="Categoria")
        self.tabela.heading("palavras", text="Palavras-Chave")
        self.tabela.heading("data", text="Data/Hora")

        self.tabela.column("id", width=40, anchor="center")
        self.tabela.column("mensagem", width=250)
        self.tabela.column("categoria", width=150)
        self.tabela.column("palavras", width=150)
        self.tabela.column("data", width=130, anchor="center")

        scrollbar = ttk.Scrollbar(frame_hist, orient="vertical", command=self.tabela.yview)
        self.tabela.configure(yscroll=scrollbar.set)
        
        self.tabela.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _processar_classificacao(self):
        texto_raw = self.txt_entrada.get("1.0", tk.END).strip()
        texto_limpo = html.escape(texto_raw)

        if not texto_limpo:
            messagebox.showwarning("Aviso", "Por favor, digite uma mensagem antes de classificar.")
            return

        # Chamada ao Módulo NLP
        resultado = classificar_mensagem(texto_limpo)
        categoria = resultado["categoria"]
        palavras_str = ", ".join(resultado["palavras_encontradas"]) if resultado["palavras_encontradas"] else "Nenhuma"

        # Salvar no SQLite via Model
        SolicitacaoModel.salvar(texto_limpo, categoria, palavras_str)

        # Atualizar a View
        self.lbl_categoria.config(text=f"Categoria: {categoria}")
        self.lbl_palavras.config(text=f"Palavras-chave: {palavras_str}")
        self.txt_entrada.delete("1.0", tk.END)

        self._carregar_historico()

    def _carregar_historico(self):
        for row in self.tabela.get_children():
            self.tabela.delete(row)

        registros = SolicitacaoModel.listar_todos()
        for reg in registros:
            self.tabela.insert("", "end", values=(
                reg["id"],
                reg["mensagem"],
                reg["categoria"],
                reg["palavras_encontradas"],
                reg["data_hora"]
            ))

# ==============================================================================
# EXECUTÁVEL PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    app = AppClassificador()
    app.mainloop()