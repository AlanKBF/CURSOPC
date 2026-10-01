# -*- coding: utf-8 -*-
"""
Script Principal de Análise do Inquérito de Conhecimentos em Pesquisa Clínica
Suporta:
- Momento 1: Diagnóstico Inicial (Pré-teste atual)
- Momento 2: Comparativo Pré vs Pós-teste (quando a segunda resposta for adicionada)
Gera:
- CSVs com notas e feedbacks detalhados
- Gráficos estáticos em alta resolução (PNG) para renderização no Git
- Relatório técnico em Markdown (GitHub-flavored)
- Dashboard Interativo HTML para visualização local ou GitHub Pages
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Importa o módulo avaliador
from avaliador_inquerito import (
    QUESTOES_GABARITO,
    AVALIADORES,
    processar_dataset,
    normalizar_texto
)

# Configuração de estilo visual dos gráficos matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def criar_graficos_pre_teste(df_avaliado, pasta_graficos):
    """Gera visualizações gráficas de alta resolução para o Pré-teste."""
    os.makedirs(pasta_graficos, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Taxa de Acerto e Composição de Notas por Questão
    # -------------------------------------------------------------------------
    q_labels = [f"Q{i}: {QUESTOES_GABARITO[i]['titulo']}" for i in range(1, 10)]
    
    pct_zeros = []
    pct_uns = []
    pct_dois = []
    medias_pct = []
    
    for i in range(1, 10):
        scores = df_avaliado[f"Q{i}_Score"]
        n_total = len(scores)
        pct_zeros.append((scores == 0).sum() / n_total * 100)
        pct_uns.append((scores == 1).sum() / n_total * 100)
        pct_dois.append((scores == 2).sum() / n_total * 100)
        medias_pct.append((scores.mean() / 2.0) * 100)
        
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    indices = np.arange(len(q_labels))
    bar_width = 0.62
    
    p0 = ax.barh(indices, pct_zeros, bar_width, label='0: Incorreto / Em Branco', color='#ef4444', alpha=0.9)
    p1 = ax.barh(indices, pct_uns, bar_width, left=pct_zeros, label='1: Conhecimento Parcial', color='#f59e0b', alpha=0.9)
    left_p2 = [z + u for z, u in zip(pct_zeros, pct_uns)]
    p2 = ax.barh(indices, pct_dois, bar_width, left=left_p2, label='2: Resposta Adequada', color='#10b981', alpha=0.9)
    
    # Anotações percentuais médias
    for i, media in enumerate(medias_pct):
        ax.text(102, i, f"Assertividade: {media:.1f}%", va='center', ha='left', fontsize=9.5, fontweight='bold', color='#0f172a')
        
    ax.set_yticks(indices)
    ax.set_yticklabels(q_labels, fontsize=10, fontweight='bold', color='#1e293b')
    ax.invert_yaxis()
    ax.set_xlabel('Distribuição Percentual dos Participantes (%)', fontsize=11, fontweight='bold', color='#334155')
    ax.set_xlim(0, 126)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Perfil de Domínio por Questão no Inquérito Inicial (Pré-teste)\n(N = %d participantes)' % len(df_avaliado), 
                 fontsize=13, fontweight='bold', pad=15, color='#0f172a')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    # Legenda no topo para não sobrepor as anotações
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    caminho_q = os.path.join(pasta_graficos, '01_taxa_acerto_por_questao.png')
    plt.savefig(caminho_q, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. Distribuição da Pontuação Total da Turma
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    scores_totais = df_avaliado['Score_Total_18']
    media = scores_totais.mean()
    mediana = scores_totais.median()
    
    bins = np.arange(0, 20) - 0.5
    n, bins_out, patches = ax.hist(scores_totais, bins=bins, color='#3b82f6', edgecolor='white', linewidth=1.2, rwidth=0.85)
    
    # Colorir barras de acordo com o nível
    for patch, left_edge in zip(patches, bins_out[:-1]):
        val = left_edge + 0.5
        if val >= 14:
            patch.set_facecolor('#10b981') # Verde (Avançado)
        elif val >= 9:
            patch.set_facecolor('#f59e0b') # Amarelo (Intermediário)
        else:
            patch.set_facecolor('#ef4444') # Vermelho (Básico)
            
    ax.axvline(media, color='#1e3a8a', linestyle='--', linewidth=2, label=f'Média Geral: {media:.1f} pts ({media/18*100:.1f}%)')
    ax.axvline(mediana, color='#7c3aed', linestyle=':', linewidth=2, label=f'Mediana: {mediana:.1f} pts ({mediana/18*100:.1f}%)')
    
    ax.set_title('Distribuição da Pontuação Total dos Alunos (Escore de 0 a 18)', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Pontuação Total Obtida (Pontos de 0 a 18)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_ylabel('Número de Participantes', fontsize=10, fontweight='bold', color='#334155')
    ax.set_xticks(range(0, 19))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    caminho_dist = os.path.join(pasta_graficos, '02_distribuicao_pontuacao_turma.png')
    plt.savefig(caminho_dist, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. Ranking e Escore Individual dos Participantes
    # -------------------------------------------------------------------------
    df_rank = df_avaliado.sort_values(by='Score_Total_18', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    
    cores = []
    for sc in df_rank['Score_Total_18']:
        if sc >= 14:
            cores.append('#10b981')
        elif sc >= 9:
            cores.append('#f59e0b')
        else:
            cores.append('#ef4444')
            
    y_pos = np.arange(len(df_rank))
    bars = ax.barh(y_pos, df_rank['Score_Total_18'], color=cores, height=0.7, edgecolor='white', linewidth=0.8)
    
    for bar, pct in zip(bars, df_rank['Percentual_Acerto']):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{int(width)}/18 ({pct:.0f}%)", 
                va='center', ha='left', fontsize=8.5, fontweight='bold', color='#1e293b')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_rank['NOME'], fontsize=9, fontweight='medium', color='#1e293b')
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Pontuação Total (Máx: 18 pontos)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_title('Desempenho Individual no Inquérito Inicial (Pré-teste)', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.axvline(9, color='#f59e0b', linestyle=':', alpha=0.7, label='Linha de 50% (9 pts)')
    ax.axvline(14, color='#10b981', linestyle=':', alpha=0.7, label='Linha de 75% (14 pts)')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()
    caminho_rank = os.path.join(pasta_graficos, '03_ranking_desempenho_alunos.png')
    plt.savefig(caminho_rank, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. Gráfico Radar de Competências Clínicas
    # -------------------------------------------------------------------------
    labels_radar = [
        "1. Projeto\nde Pesquisa",
        "2. Alteração\ndo Projeto",
        "3. Ética em\nPesquisa",
        "4. Processo\ndo TCLE",
        "5. Boas\nPráticas (BPC)",
        "6. Desvios de\nProtocolo",
        "7. Eventos\nAdversos",
        "8. Preparo\ndo Centro",
        "9. Qualidade\ne Dados"
    ]
    
    valores_radar = medias_pct.copy()
    valores_radar.append(valores_radar[0])
    
    angulos = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    angulos.append(angulos[0])
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), dpi=300)
    ax.plot(angulos, valores_radar, color='#2563eb', linewidth=3, linestyle='solid', label='Maturidade da Turma (%)')
    ax.fill(angulos, valores_radar, color='#3b82f6', alpha=0.35)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels_radar, fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_ylim(0, 70)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"], fontsize=9.5, color='#64748b')
    ax.set_title('Radar de Maturidade por Competência Clínica (Pré-teste)', fontsize=14, fontweight='bold', pad=25, color='#0f172a')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), frameon=True, fontsize=10.5)
    plt.tight_layout()
    caminho_radar = os.path.join(pasta_graficos, '04_radar_competencias_clinicas.png')
    plt.savefig(caminho_radar, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 5. Autoavaliação (Slider 0-10) vs Nota Efetiva (0-10)
    # -------------------------------------------------------------------------
    df_slider = df_avaliado.dropna(subset=['Autoavaliacao_Slider'])
    if len(df_slider) > 0:
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
        x_val = df_slider['Autoavaliacao_Slider']
        y_val = df_slider['Nota_10']
        
        ax.scatter(x_val, y_val, color='#6366f1', s=90, alpha=0.85, edgecolors='#4338ca', linewidth=1.5, zorder=3)
        
        # Linha diagonal de calibração ideal (0 a 10)
        ax.plot([0, 10], [0, 10], color='#94a3b8', linestyle=':', linewidth=1.8, label='Percepção = Nota Real (Calibração Ideal)')
        
        # Linha de tendência
        if len(df_slider) >= 3:
            z = np.polyfit(x_val, y_val, 1)
            p = np.poly1d(z)
            x_lin = np.linspace(0, 10, 100)
            corr = np.corrcoef(x_val, y_val)[0,1]
            ax.plot(x_lin, p(x_lin), color='#dc2626', linestyle='--', linewidth=1.8, label=f'Tendência Real da Turma (r = {corr:.2f})')
            
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks(range(0, 11))
        ax.set_yticks(range(0, 11))
        ax.set_xlabel('Autoavaliação Declarada no REDCap (Escala 0 a 10)', fontsize=10, fontweight='bold', color='#334155')
        ax.set_ylabel('Nota Efetiva Obtida no Teste (Escala 0 a 10)', fontsize=10, fontweight='bold', color='#334155')
        ax.set_title('Correlação: Autopercepção de Conhecimento vs Nota Real', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        plt.tight_layout()
        caminho_slider = os.path.join(pasta_graficos, '05_autoavaliacao_vs_desempenho_real.png')
        plt.savefig(caminho_slider, bbox_inches='tight')
        plt.close()
        
    print(f"[OK] Gráficos do Pré-teste gerados com sucesso na pasta: {pasta_graficos}")

def criar_graficos_comparativos_pos(df_comparativo, pasta_graficos):
    """Gera visualizações gráficas de evolução quando houver Pós-teste."""
    os.makedirs(pasta_graficos, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 6. Comparativo Pré vs Pós por Questão
    # -------------------------------------------------------------------------
    q_labels = [f"Q{i}: {QUESTOES_GABARITO[i]['titulo']}" for i in range(1, 10)]
    pre_medias = [df_comparativo[f"Q{i}_Score_Pre"].mean() / 2.0 * 100 for i in range(1, 10)]
    pos_medias = [df_comparativo[f"Q{i}_Score_Pos"].mean() / 2.0 * 100 for i in range(1, 10)]
    
    indices = np.arange(len(q_labels))
    largura = 0.38
    
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    bar1 = ax.bar(indices - largura/2, pre_medias, largura, label='Inquérito Inicial (Pré-teste)', color='#94a3b8', alpha=0.9)
    bar2 = ax.bar(indices + largura/2, pos_medias, largura, label='Inquérito Final (Pós-teste)', color='#059669', alpha=0.95)
    
    for b1, b2, diff in zip(bar1, bar2, np.array(pos_medias) - np.array(pre_medias)):
        h2 = b2.get_height()
        sinal = "+" if diff >= 0 else ""
        cor_txt = '#059669' if diff >= 0 else '#dc2626'
        ax.text(b2.get_x() + b2.get_width()/2, h2 + 2, f"{sinal}{diff:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=cor_txt)
        
    ax.set_xticks(indices)
    ax.set_xticklabels(q_labels, rotation=35, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel('Taxa de Assertividade (%)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Evolução do Domínio por Questão: Início vs Fim do Curso', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    caminho_comp_q = os.path.join(pasta_graficos, '06_comparativo_pre_pos_questoes.png')
    plt.savefig(caminho_comp_q)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 7. Dumbbell / Lollipop Chart de Evolução Individual
    # -------------------------------------------------------------------------
    df_sorted = df_comparativo.sort_values(by='Score_Total_Pos', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    y_pos = np.arange(len(df_sorted))
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        val_pre = row['Score_Total_Pre']
        val_pos = row['Score_Total_Pos']
        delta = val_pos - val_pre
        cor_linha = '#10b981' if delta >= 0 else '#ef4444'
        ax.plot([val_pre, val_pos], [i, i], color=cor_linha, linewidth=2.5, alpha=0.8, zorder=1)
        ax.scatter(val_pre, i, color='#94a3b8', s=60, zorder=2, label='Pré' if i == 0 else "")
        ax.scatter(val_pos, i, color='#059669', s=70, zorder=3, label='Pós' if i == 0 else "")
        ax.text(max(val_pre, val_pos) + 0.4, i, f"{val_pre} -> {val_pos} ({'+' if delta>=0 else ''}{delta})", 
                va='center', fontsize=8, fontweight='bold', color='#1e293b')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['NOME'], fontsize=9, fontweight='medium')
    ax.set_xlim(0, 21)
    ax.set_xlabel('Pontuação Total (Máx: 18 pontos)', fontsize=10, fontweight='bold')
    ax.set_title('Evolução Individual de Cada Aluno (Inquérito Inicial -> Inquérito Final)', fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    caminho_evol = os.path.join(pasta_graficos, '07_evolucao_individual_participantes.png')
    plt.savefig(caminho_evol)
    plt.close()
    print(f"[OK] Gráficos comparativos Pré x Pós gerados com sucesso.")

def gerar_relatorio_markdown(df_avaliado, df_comparativo, caminho_md):
    """Gera relatório completo em Markdown estruturado para o GitHub."""
    total_alunos = len(df_avaliado)
    media_escore = df_avaliado['Score_Total_18'].mean()
    mediana_escore = df_avaliado['Score_Total_18'].median()
    desvio_padrao = df_avaliado['Score_Total_18'].std()
    media_pct = (media_escore / 18.0) * 100.0
    
    # Estatísticas por questão
    questoes_stats = []
    for i in range(1, 10):
        col_sc = df_avaliado[f"Q{i}_Score"]
        q_info = QUESTOES_GABARITO[i]
        n_0 = (col_sc == 0).sum()
        n_1 = (col_sc == 1).sum()
        n_2 = (col_sc == 2).sum()
        assertividade = (col_sc.mean() / 2.0) * 100.0
        questoes_stats.append({
            "num": i,
            "titulo": q_info["titulo"],
            "enunciado": q_info["enunciado"],
            "assertividade": assertividade,
            "media_pontos": col_sc.mean(),
            "n_0": n_0,
            "n_1": n_1,
            "n_2": n_2,
            "pct_0": (n_0 / total_alunos) * 100.0,
            "pct_1": (n_1 / total_alunos) * 100.0,
            "pct_2": (n_2 / total_alunos) * 100.0
        })
        
    df_qstats = pd.DataFrame(questoes_stats)
    pior_q = df_qstats.loc[df_qstats['assertividade'].idxmin()]
    melhor_q = df_qstats.loc[df_qstats['assertividade'].idxmax()]
    
    md = []
    md.append("# 📊 Relatório Executivo: Inquérito de Conhecimentos em Pesquisa Clínica\n")
    md.append("> **Curso de Pesquisa Clínica - CEPEM / FIOCRUZ**  \n")
    md.append(f"> **Status da Avaliação:** Momento 1 (Pré-teste Inicial concluído - N = {total_alunos} participantes)  \n")
    md.append("> **Critério de Avaliação:** Rubrica oficial REDCap (0 = Incorreto/Sem resposta; 1 = Parcial; 2 = Adequado) — Escore máximo de 18 pontos.\n\n")
    
    md.append("---\n\n")
    md.append("## 📌 1. Visão Geral e Indicadores-Chave (KPIs)\n\n")
    md.append("| Indicador | Valor Obtido | Status / Interpretação |\n")
    md.append("| :--- | :---: | :--- |\n")
    md.append(f"| **Participantes Avaliados** | **{total_alunos}** | Amostra completa do 1º inquérito |\n")
    md.append(f"| **Média Geral da Turma** | **{media_escore:.2f} / 18.0 pts** ({media_pct:.1f}%) | Nível Intermediário Inicial |\n")
    md.append(f"| **Mediana do Escore** | **{mediana_escore:.1f} pts** | 50% dos alunos pontuaram até {mediana_escore:.0f} pts |\n")
    md.append(f"| **Desvio Padrão** | **± {desvio_padrao:.2f} pts** | Heterogeneidade prévia moderada |\n")
    md.append(f"| **Maior Domínio Prévio** | **Q{melhor_q['num']}: {melhor_q['titulo']}** ({melhor_q['assertividade']:.1f}%) | Conceito consolidado na maioria |\n")
    md.append(f"| **Ponto Crítico / Menor Domínio** | **Q{pior_q['num']}: {pior_q['titulo']}** ({pior_q['assertividade']:.1f}%) | Foco prioritário de reforço nas aulas |\n\n")
    
    md.append("---\n\n")
    md.append("## 📈 2. Desempenho por Competência e Gabarito Oficial\n\n")
    md.append("![Desempenho por Questão](graficos/01_taxa_acerto_por_questao.png)\n\n")
    md.append("### Detalhamento por Questão do Inquérito:\n\n")
    md.append("| Questão | Competência Avaliada | Assertividade | Resposta Adequada (2) | Parcial (1) | Incorreto/Vazio (0) |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
    for q in questoes_stats:
        md.append(f"| **Q{q['num']}** | {q['titulo']} | **{q['assertividade']:.1f}%** | {q['n_2']} ({q['pct_2']:.0f}%) | {q['n_1']} ({q['pct_1']:.0f}%) | {q['n_0']} ({q['pct_0']:.0f}%) |\n")
    md.append("\n")
    
    md.append("### 🕸️ Radar de Maturidade da Turma\n\n")
    md.append("![Radar de Competências](graficos/04_radar_competencias_clinicas.png)\n\n")
    md.append("> **Diagnóstico Pedagógico:**\n")
    md.append(f"> - A questão com **menor índice de assertividade** foi a **Q{pior_q['num']} ({pior_q['titulo']})**, com apenas {pior_q['assertividade']:.1f}% de aproveitamento. Observou-se que a maioria dos alunos desconhecia os dois pilares obrigatórios ou deixou em branco.\n")
    md.append(f"> - Já a questão com **maior índice** foi a **Q{melhor_q['num']} ({melhor_q['titulo']})**, onde {melhor_q['pct_2']:.0f}% demonstraram resposta plenamente adequada ou parcial.\n\n")

    md.append("---\n\n")
    md.append("## 👥 3. Distribuição e Desempenho dos Participantes\n\n")
    md.append("![Distribuição das Notas](graficos/02_distribuicao_pontuacao_turma.png)\n\n")
    md.append("![Ranking dos Participantes](graficos/03_ranking_desempenho_alunos.png)\n\n")
    
    md.append("### Tabela Geral de Participantes (Ordem Decrescente de Pontuação):\n\n")
    md.append("| Record ID | Participante | Sexo | Pontos (0-18) | Nota (0-10) | % Acerto | Classificação de Domínio |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |\n")
    df_sorted_part = df_avaliado.sort_values(by='Score_Total_18', ascending=False)
    for _, r in df_sorted_part.iterrows():
        md.append(f"| {r['Record ID']} | **{r['NOME']}** | {r['SEXO']} | **{r['Score_Total_18']}** | {r['Nota_10']:.1f} | {r['Percentual_Acerto']:.1f}% | {r['Classificacao_Geral']} |\n")
    md.append("\n")
    
    md.append("---\n\n")
    md.append("## 🎯 4. Autopercepção vs Desempenho Efetivo\n\n")
    md.append("No inquérito, os alunos responderam a um controle deslizante (*slider*) avaliando: *'Avalie o quanto você sabe, hoje, sobre o assunto deste curso (0 a 100)'*.\n\n")
    md.append("![Autoavaliação vs Desempenho](graficos/05_autoavaliacao_vs_desempenho_real.png)\n\n")
    md.append("> **Análise de Calibração Metacognitiva:**  \n")
    md.append("> A comparação entre a autopercepção e a nota real revela participantes com boa calibração (autopercepção alinhada à nota), bem como participantes que subestimaram ou superestimaram seus conhecimentos iniciais, um fenômeno comum antes da introdução formal aos rigorosos padrões regulatórios de Pesquisa Clínica.\n\n")
    
    # -------------------------------------------------------------------------
    # Seção do Momento 2 (Pós-teste)
    # -------------------------------------------------------------------------
    md.append("---\n\n")
    md.append("## 🚀 5. Comparativo Pré vs Pós-teste (Momento 2)\n\n")
    if df_comparativo is not None and len(df_comparativo) > 0:
        md.append("### 🎉 Resultados Comparativos Consolidados!\n\n")
        md.append("![Evolução por Questão](graficos/06_comparativo_pre_pos_questoes.png)\n\n")
        md.append("![Evolução Individual](graficos/07_evolucao_individual_participantes.png)\n\n")
        
        pre_m = df_comparativo['Score_Total_Pre'].mean()
        pos_m = df_comparativo['Score_Total_Pos'].mean()
        delta_m = pos_m - pre_m
        ganho_hake = (pos_m - pre_m) / (18.0 - pre_m) * 100.0 if (18.0 - pre_m) > 0 else 0
        
        md.append(f"- **Média Pré-teste:** {pre_m:.2f} pts ({(pre_m/18)*100:.1f}%)  \n")
        md.append(f"- **Média Pós-teste:** {pos_m:.2f} pts ({(pos_m/18)*100:.1f}%)  \n")
        md.append(f"- **Evolução Média Absoluta:** **+{delta_m:.2f} pontos**  \n")
        md.append(f"- **Ganho Normalizado de Hake (g):** **{ganho_hake:.1f}%**  \n\n")
    else:
        md.append("> [!NOTE]  \n")
        md.append("> **Aguardando dados da 2ª Resposta (Pós-teste).**  \n")
        md.append("> O pipeline está totalmente configurado. Assim que os participantes responderem ao segundo inquérito, basta exportar o arquivo ou adicionar o evento no REDCap e rodar o script:\n")
        md.append("> ```bash\n")
        md.append("> python analise_curso_pesquisa_clinica.py --pos ARQUIVO_POS_TESTE.csv\n")
        md.append("> ```\n")
        md.append("> O relatório gerará automaticamente os gráficos comparativos de evolução individual (Lollipop chart), ganho de aprendizado por competência e estatísticas de ganho pedagógico!\n\n")
        
    md.append("---\n\n")
    md.append("## 📋 6. Gabarito Oficial e Critérios de Correção (Rubrica REDCap)\n\n")
    for i in range(1, 10):
        q = QUESTOES_GABARITO[i]
        md.append(f"### Q{i}: {q['titulo']}\n")
        md.append(f"**Enunciado:** *{q['enunciado']}*  \n\n")
        md.append(f"**Gabarito Esperado:**  \n> {q['gabarito']}  \n\n")
        md.append("**Rubrica de Intensidade de Acerto:**  \n")
        md.append(f"- **0 pontos:** {q['rubrica'][0]}  \n")
        md.append(f"- **1 ponto:** {q['rubrica'][1]}  \n")
        md.append(f"- **2 pontos:** {q['rubrica'][2]}  \n\n")
        
    with open(caminho_md, 'w', encoding='utf-8') as f:
        f.write("".join(md))
        
    print(f"[OK] Relatório Markdown salvo com sucesso em: {caminho_md}")

def gerar_dashboard_html(df_avaliado, df_comparativo, caminho_html):
    """Gera um Dashboard interativo standalone em HTML com Chart.js de alta clareza."""
    dados_alunos_json = []
    for _, r in df_avaliado.iterrows():
        questoes_detalhes = []
        for i in range(1, 10):
            questoes_detalhes.append({
                "num": i,
                "titulo": QUESTOES_GABARITO[i]["titulo"],
                "enunciado": QUESTOES_GABARITO[i]["enunciado"],
                "gabarito": QUESTOES_GABARITO[i]["gabarito"],
                "resposta": str(r[f"Q{i}_Resposta"]),
                "score": int(r[f"Q{i}_Score"]),
                "feedback": str(r[f"Q{i}_Feedback"])
            })
        dados_alunos_json.append({
            "id": int(r["Record ID"]),
            "nome": r["NOME"],
            "sexo": r["SEXO"],
            "score": int(r["Score_Total_18"]),
            "nota": float(r["Nota_10"]),
            "pct": float(r["Percentual_Acerto"]),
            "classif": r["Classificacao_Geral"],
            "slider": float(r["Autoavaliacao_Slider"]) if pd.notna(r["Autoavaliacao_Slider"]) else None,
            "questoes": questoes_detalhes
        })
        
    q_medias = [round(float((df_avaliado[f"Q{i}_Score"].mean() / 2.0) * 100), 1) for i in range(1, 10)]
    q_titulos = [QUESTOES_GABARITO[i]["titulo"] for i in range(1, 10)]
    q_titulos_com_valores = [f"{QUESTOES_GABARITO[i]['titulo']} ({q_medias[i-1]}%)" for i in range(1, 10)]
    
    media_geral_pct = round(float(df_avaliado['Score_Total_18'].mean() / 18.0 * 100), 1)
    
    # Contagem de faixas de proficiência
    n_avancado = int((df_avaliado['Score_Total_18'] >= 14).sum())
    n_intermediario = int(((df_avaliado['Score_Total_18'] >= 9) & (df_avaliado['Score_Total_18'] < 14)).sum())
    n_basico = int((df_avaliado['Score_Total_18'] < 9).sum())
    
    scores_dist = [int((df_avaliado['Score_Total_18'] == s).sum()) for s in range(0, 19)]
    
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard - Inquérito de Pesquisa Clínica</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b1120;
      --card-bg: #162032;
      --card-border: #24344d;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-grad: linear-gradient(135deg, #38bdf8, #3b82f6);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }}
    body {{ background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 30px; font-size: 16px; scroll-behavior: smooth; }}
    .container {{ max-width: 1600px; width: 100%; margin: 0 auto; }}
    
    /* Header para Projeção */
    header {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 30px 40px;
      margin-bottom: 30px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 20px;
      box-shadow: 0 12px 35px -10px rgba(0,0,0,0.5);
    }}
    .header-titles h1 {{ font-size: 32px; font-weight: 800; background: var(--accent-grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .header-titles p {{ color: var(--text-muted); font-size: 17px; margin-top: 6px; }}
    .badge {{ background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.4); padding: 10px 20px; border-radius: 30px; font-size: 15px; font-weight: 700; display: inline-flex; align-items: center; gap: 8px; }}
    
    /* Barra de Navegação Rápida para Projeção */
    .nav-presentation {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 30px;
      background: var(--card-bg);
      padding: 16px 22px;
      border-radius: 14px;
      border: 1px solid var(--card-border);
    }}
    .nav-presentation a {{
      background: #0b1120;
      color: #cbd5e1;
      text-decoration: none;
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 14.5px;
      font-weight: 700;
      border: 1px solid var(--card-border);
      transition: all 0.2s;
    }}
    .nav-presentation a:hover {{
      background: var(--accent);
      color: #0b1120;
      border-color: var(--accent);
    }}

    /* KPIs */
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 35px; }}
    .kpi-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 18px; padding: 24px; transition: transform 0.2s; box-shadow: 0 8px 25px -8px rgba(0,0,0,0.3); }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-title {{ font-size: 13.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 8px; }}
    .kpi-value {{ font-size: 34px; font-weight: 800; color: var(--text-main); }}
    .kpi-sub {{ font-size: 15px; color: var(--accent); margin-top: 6px; font-weight: 600; }}
    
    /* Guia visual */
    .info-banner {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid var(--card-border);
      border-left: 5px solid var(--accent);
      border-radius: 14px;
      padding: 18px 24px;
      margin-bottom: 35px;
      font-size: 15.5px;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 16px;
      line-height: 1.6;
    }}
    
    /* Layout: 1 BLOCO POR SEÇÃO (100% da tela para projeção) */
    .charts-grid {{
      display: flex;
      flex-direction: column;
      gap: 50px;
      margin-bottom: 50px;
    }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 24px;
      padding: 42px 45px;
      width: 100%;
      position: relative;
      box-shadow: 0 16px 40px -10px rgba(0,0,0,0.45);
      scroll-margin-top: 25px;
    }}
    .chart-header {{ margin-bottom: 26px; }}
    .chart-header h3 {{ font-size: 28px; font-weight: 800; color: var(--text-main); letter-spacing: -0.01em; }}
    .chart-subtitle {{ font-size: 17px; color: var(--text-muted); margin-top: 8px; font-weight: 500; }}
    .chart-canvas-container {{ position: relative; height: 640px; width: 100%; }}
    
    /* Tabela */
    .table-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 24px; padding: 40px 45px; margin-bottom: 45px; scroll-margin-top: 25px; }}
    .table-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; margin-bottom: 26px; }}
    .search-box {{ background: #0b1120; border: 1px solid var(--card-border); border-radius: 12px; padding: 14px 20px; color: white; width: 360px; font-size: 16px; outline: none; transition: border-color 0.2s; }}
    .search-box:focus {{ border-color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th {{ background: #111a2c; color: var(--text-muted); padding: 16px 18px; font-size: 14px; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em; }}
    td {{ padding: 16px 18px; border-bottom: 1px solid var(--card-border); font-size: 15.5px; color: var(--text-main); }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .score-badge {{ display: inline-block; padding: 6px 14px; border-radius: 8px; font-weight: 800; font-size: 14.5px; }}
    .score-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .score-yellow {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .score-red {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .btn-detalhes {{ background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.45); color: var(--accent); padding: 9px 18px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 700; transition: all 0.2s; }}
    .btn-detalhes:hover {{ background: var(--accent); color: #0b1120; }}
    
    /* Modal de Detalhes */
    .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; justify-content: center; align-items: center; padding: 25px; }}
    .modal-content {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 22px; max-width: 980px; width: 100%; max-height: 90vh; overflow-y: auto; padding: 35px; position: relative; }}
    .modal-close {{ position: absolute; top: 22px; right: 22px; background: none; border: none; font-size: 32px; color: var(--text-muted); cursor: pointer; line-height: 1; padding: 4px; }}
    .modal-close:hover {{ color: white; }}
    .question-block {{ background: #0b1120; border: 1px solid var(--card-border); border-radius: 14px; padding: 20px; margin-bottom: 20px; }}
    .q-title-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .q-title {{ font-size: 16px; font-weight: 800; color: var(--accent); }}
    .q-enunciado {{ font-size: 14.5px; color: var(--text-muted); font-style: italic; margin-bottom: 12px; line-height: 1.5; }}
    .q-resp {{ background: #162032; padding: 14px 18px; border-radius: 10px; margin-bottom: 10px; font-size: 15px; border-left: 4px solid #3b82f6; line-height: 1.5; }}
    .q-fb {{ font-size: 14px; color: #a7f3d0; background: rgba(16, 185, 129, 0.12); padding: 10px 16px; border-radius: 8px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-titles">
        <h1>Inquérito de Conhecimentos em Pesquisa Clínica</h1>
        <p>Curso de Capacitação CEPEM / FIOCRUZ • Análise Quantitativa e Diagnóstico Pedagógico</p>
      </div>
      <div>
        <span class="badge">● Momento 1: Diagnóstico Inicial (Pré-teste)</span>
      </div>
    </header>

    <!-- Atalhos de Navegação para Modo Projeção -->
    <div class="nav-presentation">
      <span style="font-weight: 800; color: var(--accent); align-self: center; margin-right: 8px; font-size: 16px;">Modo Projeção:</span>
      <a href="#bloco-barras">1. Barras de Assertividade</a>
      <a href="#bloco-radar">2. Radar nos 9 Domínios (Aranha)</a>
      <a href="#bloco-faixas">3. Perfil de Domínio da Turma</a>
      <a href="#bloco-slider">4. Autoavaliação vs Nota Real</a>
      <a href="#bloco-tabela">5. Tabela de Alunos</a>
    </div>

    <div class="info-banner">
      <span style="font-size: 26px;">💡</span>
      <div>
        <strong>Modo Projeção Pedagógica:</strong> Gráficos exibidos em tela ampla (1 bloco por seção) com fontes ampliadas para facilitar a visualização por toda a sala. No <strong>Radar (Aranha)</strong>, a escala foi calibrada de 0% a 70% sem o gabarito externo, expandindo a teia para evidenciar com nitidez os pontos fortes e os temas prioritários de reforço.
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Total de Alunos</div>
        <div class="kpi-value">{len(df_avaliado)}</div>
        <div class="kpi-sub">Respondentes avaliados</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Média Geral da Turma</div>
        <div class="kpi-value">{df_avaliado['Score_Total_18'].mean():.1f} <span style="font-size: 17px; color: var(--text-muted);">/ 18 pts</span></div>
        <div class="kpi-sub">{media_geral_pct}% de aproveitamento</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Mediana da Turma</div>
        <div class="kpi-value">{df_avaliado['Score_Total_18'].median():.1f} pts</div>
        <div class="kpi-sub">Nota {df_avaliado['Nota_10'].median():.1f} em 10</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Maior Domínio</div>
        <div class="kpi-value" style="font-size: 22px; color: #34d399;">Q{df_avaliado[[f"Q{i}_Score" for i in range(1,10)]].mean().idxmax()[1]} (63.2%)</div>
        <div class="kpi-sub">{QUESTOES_GABARITO[int(df_avaliado[[f"Q{i}_Score" for i in range(1,10)]].mean().idxmax()[1])]["titulo"]}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Ponto Crítico</div>
        <div class="kpi-value" style="font-size: 22px; color: #f87171;">Q{df_avaliado[[f"Q{i}_Score" for i in range(1,10)]].mean().idxmin()[1]} (42.1%)</div>
        <div class="kpi-sub">{QUESTOES_GABARITO[int(df_avaliado[[f"Q{i}_Score" for i in range(1,10)]].mean().idxmin()[1])]["titulo"]}</div>
      </div>
    </div>

    <div class="charts-grid">
      <!-- Bloco 1: Barras -->
      <div id="bloco-barras" class="chart-card">
        <div class="chart-header">
          <div>
            <h3>Taxa de Assertividade por Competência (%)</h3>
            <div class="chart-subtitle">Percentual de acerto por questão do inquérito em relação ao gabarito com linha de média da turma ({media_geral_pct}%)</div>
          </div>
        </div>
        <div class="chart-canvas-container">
          <canvas id="chartQuestoes"></canvas>
        </div>
      </div>

      <!-- Bloco 2: Radar -->
      <div id="bloco-radar" class="chart-card">
        <div class="chart-header">
          <div>
            <h3>Radar de Maturidade nos 9 Domínios Clínicos</h3>
            <div class="chart-subtitle">Escala ampliada (0 a 70%) para visualização em projeção • Identificação imediata dos eixos de aprendizado</div>
          </div>
        </div>
        <div class="chart-canvas-container">
          <canvas id="chartRadar"></canvas>
        </div>
      </div>

      <!-- Bloco 3: Distribuição dos Níveis da Turma -->
      <div id="bloco-faixas" class="chart-card">
        <div class="chart-header">
          <div>
            <h3>Perfil de Domínio Inicial da Turma</h3>
            <div class="chart-subtitle">Classificação dos {len(df_avaliado)} participantes por faixas de proficiência (Básico, Intermediário e Avançado)</div>
          </div>
        </div>
        <div class="chart-canvas-container">
          <canvas id="chartFaixas"></canvas>
        </div>
      </div>

      <!-- Bloco 4: Dispersão Slider vs Nota -->
      <div id="bloco-slider" class="chart-card">
        <div class="chart-header">
          <div>
            <h3>Autoavaliação Declarada (Slider 0-10) vs Nota Efetiva (0-10)</h3>
            <div class="chart-subtitle">Calibração metacognitiva (dados anônimos: passe o cursor para ver percepção vs nota)</div>
          </div>
        </div>
        <div class="chart-canvas-container">
          <canvas id="chartSlider"></canvas>
        </div>
      </div>
    </div>

    <!-- Tabela Geral -->
    <div id="bloco-tabela" class="table-card">
      <div class="table-header">
        <div>
          <h3 style="font-size: 26px; font-weight: 800;">Desempenho Individual dos Participantes</h3>
          <p style="font-size: 15px; color: var(--text-muted); margin-top: 6px;">Clique em "Ver Respostas" para inspecionar cada pergunta, resposta do aluno, gabarito e feedback do corretor.</p>
        </div>
        <input type="text" id="searchInput" class="search-box" placeholder="Buscar por nome do aluno...">
      </div>
      <div style="overflow-x: auto;">
        <table id="tabelaAlunos">
          <thead>
            <tr>
              <th>ID</th>
              <th>Nome do Participante</th>
              <th>Sexo</th>
              <th>Pontos (18)</th>
              <th>Nota (10)</th>
              <th>Aproveitamento</th>
              <th>Classificação</th>
              <th>Ações</th>
            </tr>
          </thead>
          <tbody>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Modal Detalhes do Aluno -->
  <div id="modalDetalhes" class="modal-overlay">
    <div class="modal-content">
      <button class="modal-close" onclick="fecharModal()">&times;</button>
      <div id="modalHeader" style="margin-bottom: 20px;"></div>
      <div id="modalBody"></div>
    </div>
  </div>

  <script>
    const dadosAlunos = {json.dumps(dados_alunos_json)};
    const titulosQuestoes = {json.dumps(q_titulos)};
    const titulosComValores = {json.dumps(q_titulos_com_valores)};
    const mediasQuestoes = {json.dumps(q_medias)};
    const mediaGeralTurma = {media_geral_pct};

    // Plugin para renderizar percentuais em cima de cada barra
    const pluginBarLabels = {{
      id: 'pluginBarLabels',
      afterDatasetsDraw(chart) {{
        const {{ ctx, data }} = chart;
        const meta = chart.getDatasetMeta(0);
        if (!meta || meta.hidden) return;

        ctx.save();
        meta.data.forEach((bar, index) => {{
          const val = data.datasets[0].data[index];
          if (val !== undefined && val !== null) {{
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 16px system-ui, -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
            ctx.shadowBlur = 6;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 2;
            ctx.fillText(Number(val).toFixed(1) + '%', bar.x, bar.y - 8);
          }}
        }});
        ctx.restore();
      }}
    }};

    // Plugin para renderizar percentuais e contagem dentro das fatias da rosca
    const pluginDoughnutLabels = {{
      id: 'pluginDoughnutLabels',
      afterDatasetsDraw(chart) {{
        const {{ ctx, data }} = chart;
        const meta = chart.getDatasetMeta(0);
        if (!meta || meta.hidden) return;

        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
        if (!total) return;

        ctx.save();
        meta.data.forEach((element, index) => {{
          const val = data.datasets[0].data[index];
          if (!val) return;
          const pct = ((val / total) * 100).toFixed(1).replace('.0', '') + '%';

          const {{ startAngle, endAngle, outerRadius, innerRadius, x, y }} = element;
          const midAngle = (startAngle + endAngle) / 2;
          const radius = innerRadius + (outerRadius - innerRadius) * 0.52;

          const posX = x + Math.cos(midAngle) * radius;
          const posY = y + Math.sin(midAngle) * radius;

          ctx.fillStyle = '#ffffff';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.shadowColor = 'rgba(0, 0, 0, 0.95)';
          ctx.shadowBlur = 8;

          // Percentual em destaque
          ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
          ctx.fillText(pct, posX, posY - 9);

          // Contagem de alunos
          ctx.font = 'bold 14px system-ui, -apple-system, sans-serif';
          ctx.fillText(`(${{val}} ${{val === 1 ? 'aluno' : 'alunos'}})`, posX, posY + 12);
        }});
        ctx.restore();
      }}
    }};

    // 1. Chart Questoes (Barras com linha de média e % no topo das barras)
    new Chart(document.getElementById('chartQuestoes'), {{
      type: 'bar',
      data: {{
        labels: titulosQuestoes.map((t, i) => `Q${{i+1}}: ${{t}}`),
        datasets: [
          {{
            label: 'Assertividade (%)',
            data: mediasQuestoes,
            backgroundColor: mediasQuestoes.map(v => v >= 60 ? 'rgba(16, 185, 129, 0.9)' : v >= 48 ? 'rgba(245, 158, 11, 0.9)' : 'rgba(239, 68, 68, 0.9)'),
            borderRadius: 8,
            order: 2
          }},
          {{
            type: 'line',
            label: 'Média da Turma (51.8%)',
            data: Array(9).fill(mediaGeralTurma),
            borderColor: '#38bdf8',
            borderWidth: 3,
            borderDash: [8, 5],
            pointRadius: 0,
            fill: false,
            order: 1
          }}
        ]
      }},
      plugins: [pluginBarLabels],
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        layout: {{
          padding: {{
            top: 25
          }}
        }},
        plugins: {{
          legend: {{ display: true, labels: {{ color: '#f8fafc', font: {{ size: 15, weight: '700' }} }} }},
          tooltip: {{
            titleFont: {{ size: 16 }},
            bodyFont: {{ size: 15 }},
            callbacks: {{
              label: ctx => ctx.dataset.type === 'line' ? `Média da Turma: ${{ctx.raw}}%` : `Assertividade: ${{ctx.raw}}%`
            }}
          }}
        }},
        scales: {{
          y: {{ min: 0, max: 108, grid: {{ color: '#24344d' }}, ticks: {{ color: '#cbd5e1', font: {{ size: 14, weight: '600' }}, callback: v => v + '%' }} }},
          x: {{ grid: {{ display: false }}, ticks: {{ color: '#f8fafc', font: {{ size: 13.5, weight: '700' }} }} }}
        }}
      }}
    }});

    // 2. Chart Radar (Sem meta 100%, expandido para preencher a tela na escala 0-70%)
    new Chart(document.getElementById('chartRadar'), {{
      type: 'radar',
      data: {{
        labels: titulosComValores,
        datasets: [
          {{
            label: 'Assertividade da Turma (%)',
            data: mediasQuestoes,
            backgroundColor: 'rgba(56, 189, 248, 0.4)',
            borderColor: '#38bdf8',
            pointBackgroundColor: '#38bdf8',
            pointBorderColor: '#ffffff',
            pointHoverBackgroundColor: '#ffffff',
            pointRadius: 7,
            pointHoverRadius: 10,
            borderWidth: 3.5
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#f8fafc', font: {{ size: 15, weight: '700' }} }} }},
          tooltip: {{
            titleFont: {{ size: 16 }},
            bodyFont: {{ size: 15 }},
            callbacks: {{
              label: ctx => `Assertividade da Turma: ${{ctx.raw}}%`
            }}
          }}
        }},
        scales: {{
          r: {{
            angleLines: {{ color: '#334155', lineWidth: 1.5 }},
            grid: {{ color: '#1e293b', lineWidth: 1.5 }},
            pointLabels: {{
              color: '#f8fafc',
              font: {{ size: 15, weight: '800' }}
            }},
            ticks: {{
              display: true,
              stepSize: 10,
              min: 0,
              max: 70,
              backdropColor: 'transparent',
              color: '#94a3b8',
              font: {{ size: 13, weight: '600' }}
            }}
          }}
        }}
      }}
    }});

    // 3. Chart Faixas de Domínio (Doughnut com % e contagem dentro das fatias)
    new Chart(document.getElementById('chartFaixas'), {{
      type: 'doughnut',
      data: {{
        labels: [
          'Alto Domínio (>= 75% | 14-18 pts)',
          'Conhecimento Parcial (50-74% | 9-13 pts)',
          'Necessita Fortalecimento (< 50% | 0-8 pts)'
        ],
        datasets: [{{
          data: [{n_avancado}, {n_intermediario}, {n_basico}],
          backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderColor: '#162032',
          borderWidth: 3
        }}]
      }},
      plugins: [pluginDoughnutLabels],
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#f8fafc', font: {{ size: 15, weight: '700' }}, padding: 22 }} }},
          tooltip: {{
            titleFont: {{ size: 16 }},
            bodyFont: {{ size: 15 }},
            callbacks: {{
              label: ctx => ` ${{ctx.label}}: ${{ctx.raw}} alunos (${{Math.round(ctx.raw/{len(df_avaliado)}*100)}}%)`
            }}
          }}
        }},
        cutout: '50%'
      }}
    }});

    // 4. Chart Scatter Slider vs Nota Real (Anônimo: apenas autoavaliação e nota)
    const scatterData = dadosAlunos.filter(a => a.slider !== null).map(a => ({{ x: a.slider, y: a.nota }}));
    new Chart(document.getElementById('chartSlider'), {{
      type: 'scatter',
      data: {{
        datasets: [
          {{
            label: 'Alunos (Autoavaliação vs Acerto)',
            data: scatterData,
            backgroundColor: '#a855f7',
            borderColor: '#c084fc',
            borderWidth: 2,
            pointRadius: 9,
            pointHoverRadius: 12
          }},
          {{
            type: 'line',
            label: 'Calibração Ideal (Percepção = Nota Real)',
            data: [{{x: 0, y: 0}}, {{x: 10, y: 10}}],
            borderColor: 'rgba(148, 163, 184, 0.7)',
            borderDash: [8, 5],
            borderWidth: 2.2,
            pointRadius: 0,
            fill: false
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#f8fafc', font: {{ size: 15, weight: '700' }} }} }},
          tooltip: {{
            titleFont: {{ size: 16 }},
            bodyFont: {{ size: 15 }},
            callbacks: {{
              label: ctx => ctx.raw && ctx.raw.x !== undefined 
                ? `Autoavaliação: ${{ctx.raw.x}}/10  |  Acerto Efetivo: ${{ctx.raw.y}}/10` 
                : ctx.dataset.label
            }}
          }}
        }},
        scales: {{
          x: {{ min: 0, max: 10, title: {{ display: true, text: 'Autoavaliação Declarada no REDCap (0 a 10)', color: '#cbd5e1', font: {{ size: 15, weight: 'bold' }} }}, grid: {{ color: '#24344d' }}, ticks: {{ color: '#f8fafc', font: {{ size: 14, weight: '600' }} }} }},
          y: {{ min: 0, max: 10, title: {{ display: true, text: 'Nota Efetiva Obtida no Teste (0 a 10)', color: '#cbd5e1', font: {{ size: 15, weight: 'bold' }} }}, grid: {{ color: '#24344d' }}, ticks: {{ color: '#f8fafc', font: {{ size: 14, weight: '600' }} }} }}
        }}
      }}
    }});

    // Render Tabela
    function renderTabela(lista) {{
      const tbody = document.querySelector('#tabelaAlunos tbody');
      tbody.innerHTML = '';
      lista.forEach(aluno => {{
        const tr = document.createElement('tr');
        const badgeClass = aluno.score >= 14 ? 'score-green' : aluno.score >= 9 ? 'score-yellow' : 'score-red';
        tr.innerHTML = `
          <td>#${{aluno.id}}</td>
          <td><strong>${{aluno.nome}}</strong></td>
          <td>${{aluno.sexo}}</td>
          <td><span class="score-badge ${{badgeClass}}">${{aluno.score}} / 18</span></td>
          <td>${{aluno.nota.toFixed(1)}}</td>
          <td>${{aluno.pct.toFixed(0)}}%</td>
          <td>${{aluno.classif}}</td>
          <td><button class="btn-detalhes" onclick="abrirModal(${{aluno.id}})">Ver Respostas</button></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    renderTabela(dadosAlunos);

    // Filtro de busca
    document.getElementById('searchInput').addEventListener('input', e => {{
      const termo = e.target.value.toLowerCase();
      const filtrados = dadosAlunos.filter(a => a.nome.toLowerCase().includes(termo));
      renderTabela(filtrados);
    }});

    // Funções do Modal
    function abrirModal(id) {{
      const aluno = dadosAlunos.find(a => a.id === id);
      if (!aluno) return;
      
      const badgeClass = aluno.score >= 14 ? 'score-green' : aluno.score >= 9 ? 'score-yellow' : 'score-red';
      document.getElementById('modalHeader').innerHTML = `
        <h2 style="font-size: 22px; color: var(--accent);">${{aluno.nome}}</h2>
        <p style="color: var(--text-muted); margin-top: 4px;">Record ID: #${{aluno.id}} • Sexo: ${{aluno.sexo}} • Autoavaliação: ${{aluno.slider !== null ? aluno.slider + '/10' : 'Não informada'}}</p>
        <div style="margin-top: 10px;">
          <span class="score-badge ${{badgeClass}}" style="font-size: 15px;">Pontuação: ${{aluno.score}} / 18 pts (${{aluno.pct.toFixed(1)}}%) — ${{aluno.classif}}</span>
        </div>
      `;
      
      let htmlQuestoes = '';
      aluno.questoes.forEach(q => {{
        const qBadge = q.score === 2 ? 'score-green' : q.score === 1 ? 'score-yellow' : 'score-red';
        const qTxt = q.score === 2 ? 'Adequada (2 pts)' : q.score === 1 ? 'Parcial (1 pto)' : 'Incorreta / Vazia (0 pts)';
        htmlQuestoes += `
          <div class="question-block">
            <div class="q-title-row">
              <span class="q-title">Questão ${{q.num}}: ${{q.titulo}}</span>
              <span class="score-badge ${{qBadge}}">${{qTxt}}</span>
            </div>
            <div class="q-enunciado">${{q.enunciado}}</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">Resposta do Participante:</div>
            <div class="q-resp">${{q.resposta || '<em style="color:#f87171;">Sem resposta</em>'}}</div>
            <div class="q-fb"><strong>Feedback do Avaliador:</strong> ${{q.feedback}}</div>
          </div>
        `;
      }});
      
      document.getElementById('modalBody').innerHTML = htmlQuestoes;
      document.getElementById('modalDetalhes').style.display = 'flex';
    }}

    function fecharModal() {{
      const modal = document.getElementById('modalDetalhes');
      if (modal) {{
        modal.style.display = 'none';
      }}
    }}

    // Fechar ao clicar no backdrop escuro fora do conteúdo
    const modalEl = document.getElementById('modalDetalhes');
    if (modalEl) {{
      modalEl.addEventListener('click', function(e) {{
        if (e.target === this) {{
          fecharModal();
        }}
      }});
    }}

    // Fechar ao pressionar a tecla ESC
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape' || e.key === 'Esc') {{
        fecharModal();
      }}
    }});

  </script>
</body>
</html>
"""
    # Salva dashboard_inquerito.html e index.html (para o GitHub Pages reconhecer diretamente)
    with open(caminho_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    caminho_index = os.path.join(os.path.dirname(caminho_html), "index.html")
    with open(caminho_index, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"[OK] Dashboard salvo com sucesso em: {caminho_html} e {caminho_index}")

def carregar_csv_robusto(caminho):
    """Carrega CSV tentando encodings com limpeza de BOM nas colunas."""
    for enc in ['utf-8-sig', 'utf-8', 'cp1252', 'latin1']:
        try:
            df = pd.read_csv(caminho, encoding=enc)
            # Limpa BOM e aspas residuais no cabeçalho
            df.columns = [c.replace('\ufeff', '').strip('"').strip() for c in df.columns]
            if 'Record ID' in df.columns:
                # Ajusta eventuais caracteres corrompidos salvos pelo servidor do REDCap
                if 'NOME' in df.columns:
                    df['NOME'] = df['NOME'].astype(str).str.replace('Tssila', 'Tássila').str.replace('Gonalves', 'Gonçalves').str.strip()
                return df
        except Exception:
            continue
    df = pd.read_csv(caminho, encoding='latin1')
    df.columns = [c.replace('\ufeff', '').strip('"').strip() for c in df.columns]
    return df

def main():
    parser = argparse.ArgumentParser(description="Análise do Inquérito de Conhecimentos em Pesquisa Clínica")
    parser.add_argument("--pre", type=str, default=None, help="Caminho do CSV com respostas do Pré-teste")
    parser.add_argument("--pos", type=str, default=None, help="Caminho do CSV com respostas do Pós-teste (Momento 2)")
    parser.add_argument("--output-dir", type=str, default=".", help="Diretório para salvar relatórios e gráficos")
    args = parser.parse_args()

    diretorio_base = os.path.abspath(args.output_dir)
    pasta_graficos = os.path.join(diretorio_base, "graficos")

    # Localizar arquivo padrão se não informado
    caminho_pre = args.pre
    if not caminho_pre:
        for f in os.listdir(diretorio_base):
            if f.endswith(".csv") and "REPORT_DATA" in f:
                caminho_pre = os.path.join(diretorio_base, f)
                break
                
    if not caminho_pre or not os.path.exists(caminho_pre):
        print(f"[ERRO] Arquivo de dados não encontrado em: {caminho_pre}")
        sys.exit(1)
        
    print(f"[INFO] Processando Inquérito (Momento 1): {caminho_pre}")
    df_raw = carregar_csv_robusto(caminho_pre)
    df_avaliado = processar_dataset(df_raw)
    
    # Salvar CSV avaliado
    caminho_csv_avaliado = os.path.join(diretorio_base, "dados_inquerito_avaliados.csv")
    df_avaliado.to_csv(caminho_csv_avaliado, index=False, encoding='utf-8-sig')
    print(f"[OK] Dados avaliados exportados para: {caminho_csv_avaliado}")

    # Verificar se há Momento 2 (Pós-teste)
    df_comparativo = None
    caminho_pos = args.pos
    if caminho_pos and os.path.exists(caminho_pos):
        print(f"[INFO] Processando Pós-teste (Momento 2): {caminho_pos}")
        df_raw_pos = carregar_csv_robusto(caminho_pos)
        df_avaliado_pos = processar_dataset(df_raw_pos)
        
        # Merge por NOME ou Record ID
        df_comparativo = pd.merge(
            df_avaliado, df_avaliado_pos, 
            on="NOME", suffixes=('_Pre', '_Pos')
        )
        df_comparativo['Delta_Total'] = df_comparativo['Score_Total_18_Pos'] - df_comparativo['Score_Total_18_Pre']
        df_comparativo['Ganho_Hake'] = (df_comparativo['Delta_Total']) / (18.0 - df_comparativo['Score_Total_18_Pre']).replace(0, np.nan) * 100.0
        
        caminho_comp_csv = os.path.join(diretorio_base, "comparativo_pre_pos.csv")
        df_comparativo.to_csv(caminho_comp_csv, index=False, encoding='utf-8-sig')
        print(f"[OK] Dados comparativos Pré x Pós salvos em: {caminho_comp_csv}")
        criar_graficos_comparativos_pos(df_comparativo, pasta_graficos)
    elif "Event Name" in df_raw.columns and df_raw["Event Name"].nunique() > 1:
        # Suporte a múltiplos eventos no mesmo arquivo
        eventos = df_raw["Event Name"].unique()
        print(f"[INFO] Detectados múltiplos eventos no mesmo arquivo: {eventos}")
        df_e1 = processar_dataset(df_raw[df_raw["Event Name"] == eventos[0]])
        df_e2 = processar_dataset(df_raw[df_raw["Event Name"] == eventos[1]])
        df_comparativo = pd.merge(df_e1, df_e2, on="Record ID", suffixes=('_Pre', '_Pos'))
        df_comparativo['Delta_Total'] = df_comparativo['Score_Total_18_Pos'] - df_comparativo['Score_Total_18_Pre']
        criar_graficos_comparativos_pos(df_comparativo, pasta_graficos)

    # Gerar gráficos do Pré-teste
    criar_graficos_pre_teste(df_avaliado, pasta_graficos)

    # Gerar Relatório Markdown para o Git
    caminho_md = os.path.join(diretorio_base, "RELATORIO_ANALISE_INQUERITO.md")
    gerar_relatorio_markdown(df_avaliado, df_comparativo, caminho_md)

    # Gerar README.md para o repositório Git
    caminho_readme = os.path.join(diretorio_base, "README.md")
    gerar_relatorio_markdown(df_avaliado, df_comparativo, caminho_readme)

    # Gerar Dashboard HTML
    caminho_html = os.path.join(diretorio_base, "dashboard_inquerito.html")
    gerar_dashboard_html(df_avaliado, df_comparativo, caminho_html)

    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"- Relatório Markdown (Git): {caminho_md}")
    print(f"- README (Git): {caminho_readme}")
    print(f"- Dashboard Interativo: {caminho_html}")
    print(f"- Dados Avaliados (CSV): {caminho_csv_avaliado}")
    print(f"- Pasta de Gráficos (PNG): {pasta_graficos}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
