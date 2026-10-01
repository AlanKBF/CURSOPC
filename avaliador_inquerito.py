# -*- coding: utf-8 -*-
"""
Avaliador do Inquérito de Conhecimentos Prévios em Pesquisa Clínica
Baseado no gabarito oficial e critérios de intensidade de acerto do REDCAP.pdf:
  0 = Não respondeu / Incorreto
  1 = Conhecimento parcial
  2 = Resposta adequada
Escore total: 0 a 18 pontos (normalizado também para 0 a 100% e escala 0 a 10)
"""

import re
import unicodedata
import pandas as pd
import numpy as np

# Metadados e gabarito oficial das questões
QUESTOES_GABARITO = {
    1: {
        "titulo": "Projeto de pesquisa",
        "enunciado": "1. Cite quatro elementos essenciais que devem constar em um projeto ou protocolo de pesquisa.",
        "gabarito": "Projeto completo, TCLE/TALE, anuência das instituições envolvidas e declarações do pesquisador (ou elementos estruturais: justificativa, objetivos, metodologia, riscos/benefícios, critérios de elegibilidade, cronograma, orçamento).",
        "rubrica": {
            0: "Não respondeu, resposta incorreta ou menos de 2 elementos citados.",
            1: "Conhecimento parcial: citou 2 ou 3 elementos válidos.",
            2: "Resposta adequada: citou 4 ou mais elementos essenciais."
        }
    },
    2: {
        "titulo": "Alteração do projeto",
        "enunciado": "2. Um procedimento de um projeto já aprovado precisa ser alterado. Qual deve ser a conduta do pesquisador antes de implementar a alteração?",
        "gabarito": "Documentar toda a alteração, submeter ao CEP uma emenda ao projeto e, após aprovação ética, treinar a equipe e implementar as mudanças.",
        "rubrica": {
            0: "Não respondeu, resposta incorreta ou propôs implementar sem aprovação do CEP.",
            1: "Conhecimento parcial: citou submeter ao CEP/emenda sem citar aprovação prévia/documentação, ou apenas preparo interno sem CEP.",
            2: "Resposta adequada: indicou submissão formal de emenda ao CEP, aguardar aprovação ética e documentar/treinar antes de implementar."
        }
    },
    3: {
        "titulo": "Ética em pesquisa",
        "enunciado": "3. Em que momento uma pesquisa envolvendo seres humanos deve ser submetida à apreciação ética e por quê?",
        "gabarito": "Antes do início de qualquer atividade/procedimento do projeto. Por quê: garantir a proteção dos direitos, dignidade, integridade e bem-estar dos participantes e cumprimento das normas éticas.",
        "rubrica": {
            0: "Não respondeu ou indicou momento incorreto (durante ou após início).",
            1: "Conhecimento parcial: citou o momento correto (antes do início) mas não justificou, ou justificou sem definir o momento prévio.",
            2: "Resposta adequada: indicou o momento prévio correto (antes de qualquer atividade) E justificou com a proteção/ética dos participantes."
        }
    },
    4: {
        "titulo": "Consentimento Livre e Esclarecido",
        "enunciado": "4. Em uma frase, qual é a principal finalidade do processo de consentimento livre e esclarecido?",
        "gabarito": "Garantir que o participante decida de forma livre, autônoma e esclarecida sobre sua participação na pesquisa, após receber e compreender todas as informações necessárias.",
        "rubrica": {
            0: "Não respondeu ou definição incorreta/incompatível.",
            1: "Conhecimento parcial: abordou apenas o dever de informar/esclarecer ou apenas o ato de coletar assinatura/autorização sem enfatizar a autonomia voluntária.",
            2: "Resposta adequada: contemplou expressamente a autonomia/decisão livre e voluntária e a compreensão/esclarecimento das informações."
        }
    },
    5: {
        "titulo": "Boas Práticas Clínicas (BPC)",
        "enunciado": "5. Cite dois objetivos fundamentais das Boas Práticas na condução de uma pesquisa clínica.",
        "gabarito": "1) Proteger os direitos, a segurança e o bem-estar dos participantes da pesquisa. 2) Assegurar a qualidade, integridade, rastreabilidade e confiabilidade dos dados e resultados.",
        "rubrica": {
            0: "Não respondeu ou citou objetivos alheios às BPC.",
            1: "Conhecimento parcial: abordou apenas 1 dos 2 pilares (somente proteção dos participantes OU somente qualidade/integridade dos dados).",
            2: "Resposta adequada: abordou claramente os 2 pilares centrais das BPC (proteção dos sujeitos E qualidade/confiabilidade dos dados)."
        }
    },
    6: {
        "titulo": "Desvio do protocolo",
        "enunciado": "6. A equipe identificou que um procedimento não foi realizado conforme previsto no protocolo. Cite duas ações que devem ser tomadas a partir dessa identificação.",
        "gabarito": "1) Adotar ações corretivas e preventivas (CAPA) para mitigar o risco de recorrência. 2) Realizar o registro formal e as comunicações/notificações aplicáveis (ao CEP, patrocinador, coordenação).",
        "rubrica": {
            0: "Não respondeu ou ações inadequadas.",
            1: "Conhecimento parcial: citou apenas o registro/notificação ou apenas a correção/prevenção.",
            2: "Resposta adequada: citou ambas as ações fundamentais (registro/notificação regulatória E ação corretiva/preventiva/mitigadora)."
        }
    },
    7: {
        "titulo": "Evento adverso",
        "enunciado": "7. Cite duas ações que devem ser adotadas pela equipe ao identificar um evento adverso durante a condução de um estudo clínico.",
        "gabarito": "1) Prestar assistência necessária imediata ao participante, garantindo sua segurança, manejo clínico e acompanhamento adequado. 2) Registrar e comunicar/notificar ao patrocinador/CEP/autoridades regulatórias dentro dos prazos vigentes.",
        "rubrica": {
            0: "Não respondeu ou conduta incorreta.",
            1: "Conhecimento parcial: citou apenas a assistência clínica ou apenas o registro/notificação.",
            2: "Resposta adequada: citou a assistência/segurança do paciente E a notificação/registro aos responsáveis."
        }
    },
    8: {
        "titulo": "Preparação do centro",
        "enunciado": "8. Cite três aspectos que devem ser avaliados para determinar se um centro está preparado para conduzir um estudo clínico.",
        "gabarito": "1) Equipe qualificada e treinada. 2) Infraestrutura, instalações e recursos adequados (laboratório, farmácia, equipamentos). 3) Capacidade de recrutamento e retenção de participantes (população elegível no cronograma previsto).",
        "rubrica": {
            0: "Não respondeu ou nenhum aspecto pertinente.",
            1: "Conhecimento parcial: citou 1 ou 2 aspectos válidos.",
            2: "Resposta adequada: citou 3 ou mais aspectos estruturantes (equipe, estrutura física e recrutamento/população/qualidade)."
        }
    },
    9: {
        "titulo": "Dados e documentação",
        "enunciado": "9. Cite dois cuidados essenciais para garantir qualidade e rastreabilidade dos dados e documentos de uma pesquisa.",
        "gabarito": "1) Registrar os dados de forma completa, precisa, legível e contemporânea (no momento adequado), com autoria identificada (princípios ALCOA). 2) Manter os documentos organizados, armazenados de forma segura, com controle de acesso, sigilo e rastreabilidade de versões.",
        "rubrica": {
            0: "Não respondeu ou resposta genérica/incorreta.",
            1: "Conhecimento parcial: citou apenas 1 cuidado válido.",
            2: "Resposta adequada: citou 2 ou mais cuidados consistentes (registro contemporâneo/preciso + segurança/controle de acesso/organização)."
        }
    }
}

def normalizar_texto(texto):
    """Normaliza o texto removendo acentos e espaços supérfluos."""
    if not isinstance(texto, str) or pd.isna(texto):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'[\r\n]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

def contem_termos(texto_norm, termos):
    """Verifica se pelo menos um dos termos está no texto normalizado."""
    for t in termos:
        t_norm = normalizar_texto(t)
        if re.search(r'\b' + re.escape(t_norm) + r'\b', texto_norm) or t_norm in texto_norm:
            return True
    return False

# ==============================================================================
# FUNÇÕES DE AVALIAÇÃO INDIVIDUAL POR QUESTÃO
# ==============================================================================

def avaliar_q1(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nunca fiz', 'sem resposta', 'nan']:
        return 0, "Sem resposta ou não soube responder."
    
    catalogo = {
        "Objetivos": ['objetivo', 'objetivos'],
        "Metodologia/Método": ['metodologia', 'metodo', 'desenho', 'delineamento'],
        "TCLE/Aspectos Éticos": ['tcle', 'tale', 'termo de consentimento', 'consentimento', 'aspectos eticos', 'etica', 'cep'],
        "Justificativa/Introdução/Problema": ['justificativa', 'introducao', 'resumo', 'problema', 'tema', 'hipotese', 'referencial'],
        "Critérios de Elegibilidade/Público": ['criterios de inclusao', 'inclusao e exclusao', 'criterios', 'publico-alvo', 'populacao', 'amostra'],
        "Riscos e Benefícios": ['riscos e beneficios', 'riscos', 'beneficios'],
        "Cronograma": ['cronograma', 'prazos'],
        "Orçamento/Financiamento": ['orcamento', 'recursos financeiros', 'financiamento', 'patrocinador'],
        "Documentos Regulatórios/Institucionais": ['anuencia', 'declaracao', 'folha de rosto', 'termo de confidencialidade', 'membros', 'pops', 'delegacao']
    }
    
    elementos = [nome for nome, termos in catalogo.items() if contem_termos(t, termos)]
    qtd = len(elementos)
    
    if qtd >= 4:
        return 2, f"Adequado: identificou {qtd} elementos essenciais ({', '.join(elementos)})."
    elif qtd >= 2:
        return 1, f"Parcial: citou {qtd} elementos ({', '.join(elementos)}). O gabarito solicita quatro elementos essenciais."
    else:
        return 0, f"Insuficiente: citou apenas {qtd} elemento(s). Necessário elencar quatro componentes estruturais."

def avaliar_q2(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nunca fiz', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    tem_cep = contem_termos(t, ['cep', 'comite de etica', 'comite', 'plataforma brasil', 'emenda'])
    tem_aprovacao = contem_termos(t, ['aprovacao', 'aprovar', 'aguardar', 'apos aprovacao', 'antes de implementar', 'antes de colocar em pratica', 'nao colocar em pratica'])
    tem_doc_treino = contem_termos(t, ['documentar', 'documentacao', 'treinar', 'treinamento', 'capacitacao', 'revisar', 'detalhar', 'justificar', 'protocolar'])
    
    if tem_cep and (tem_aprovacao or tem_doc_treino):
        return 2, "Adequado: contemplou a submissão de emenda ao CEP, aguardar aprovação e preparo prévio (documentação/treinamento)."
    elif tem_cep:
        return 1, "Parcial: mencionou o CEP/emenda, mas omitiu a exigência de aguardar a aprovação formal e treinar a equipe antes da implementação."
    elif tem_doc_treino:
        return 1, "Parcial: abordou a documentação/treinamento interno, mas omitiu a submissão regulatória obrigatória de emenda ao CEP."
    else:
        return 0, "Incorreto: não mencionou a submissão de emenda ao CEP nem trâmites éticos regulatórios."

def avaliar_q3(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    momento_correto = contem_termos(t, ['antes', 'previa', 'inicio', 'antes de iniciar', 'antes de comecar', 'antes de qualquer', 'antes de lidar'])
    momento_errado = contem_termos(t, ['durante', 'depois', 'apos o inicio', 'apos iniciar'])
    
    tem_justificativa = contem_termos(t, ['proteger', 'protecao', 'seguranca', 'bem estar', 'direitos', 'dignidade', 
                                          'riscos', 'beneficios', 'principios eticos', 'normas eticas', 'danos', 'avaliar o projeto', 'dano'])
    
    if momento_correto and not momento_errado and tem_justificativa:
        return 2, "Adequado: apontou o momento correto (antes do início de qualquer atividade) e justificou com a proteção ética/segurança dos participantes."
    elif momento_correto and not momento_errado:
        return 1, "Parcial: indicou o momento temporal correto (antes do início), mas não justificou o porquê."
    elif tem_justificativa:
        return 1, "Parcial: apresentou a justificativa ética de proteção aos participantes, mas não explicitou que deve ser anterior ao início das atividades."
    else:
        return 0, "Incorreto: não indicou o momento temporal prévio exigido pelas resoluções éticas vigentes."

def avaliar_q4(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    tem_autonomia = contem_termos(t, ['livre', 'autonoma', 'autonomia', 'voluntaria', 'voluntario', 'escolha', 'desistencia', 'opcional', 'decida', 'decisao'])
    tem_esclarecimento = contem_termos(t, ['esclarec', 'inform', 'compreend', 'entendid', 'conhecimento', 'ciencia', 'riscos', 'beneficios', 'objetivos'])
    
    if tem_autonomia and tem_esclarecimento:
        return 2, "Adequado: contemplou tanto a autonomia/livre decisão voluntária quanto o pleno esclarecimento e compreensão das informações."
    elif tem_autonomia or tem_esclarecimento:
        return 1, "Parcial: contemplou apenas um dos eixos (esclarecimento da informação OU autonomia de participação)."
    else:
        return 0, "Insuficiente: não expressou a finalidade ética central do consentimento livre e esclarecido."

def avaliar_q5(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    pilar_participante = contem_termos(t, ['participante', 'paciente', 'seguranca', 'direitos', 'bem estar', 'bem-estar', 'vida', 'protecao'])
    pilar_dados = contem_termos(t, ['dados', 'qualidade', 'integridade', 'rastreabilidade', 'confiabilidade', 'padronizacao', 'resultados', 'reprodutibilidade', 'rigor', 'execucao correta'])
    
    if pilar_participante and pilar_dados:
        return 2, "Adequado: contemplou os dois pilares das BPC (proteção do participante e integridade/qualidade dos dados da pesquisa)."
    elif pilar_participante or pilar_dados:
        return 1, "Parcial: contemplou apenas um dos dois pilares fundamentais das Boas Práticas Clínicas."
    else:
        return 0, "Incorreto: não identificou os objetivos essenciais preconizados pelo ICH-GCP / Boas Práticas Clínicas."

def avaliar_q6(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    tem_registro_notif = contem_termos(t, ['registrar', 'registro', 'relatar', 'notificar', 'notificacao', 'comunicar', 'documentar', 'informar', 'aberto uma nao conformidade', 'nao conformidade'])
    tem_correcao_prev = contem_termos(t, ['corretiva', 'preventiva', 'mitigar', 'evitar', 'reincidencia', 'repeticao', 'reparadora', 'rever', 'revisar', 'capacitacao', 'adaptac', 'recomecar', 'adequac', 'corrigir', 'minimizar', 'refazer o teste'])
    
    if tem_registro_notif and tem_correcao_prev:
        return 2, "Adequado: citou registro/notificação formal do desvio e adoção de ações corretivas/preventivas para evitar reincidência."
    elif tem_registro_notif or tem_correcao_prev:
        return 1, "Parcial: citou apenas uma das condutas (registro/notificação OU ação corretiva/preventiva)."
    else:
        return 0, "Incorreto: condutas não condizentes com as Boas Práticas Clínicas para gestão de desvios."

def avaliar_q7(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    tem_assistencia = contem_termos(t, ['assistencia', 'cuidado', 'atendimento', 'seguranca', 'bem estar', 'manejo', 'acompanhamento', 'socorrer', 'tratar', 'priorizar o bem estar', 'parar a atividade', 'interromper', 'assistencia necessaria'])
    tem_registro_notif = contem_termos(t, ['notificar', 'notificacao', 'registrar', 'registro', 'comunicar', 'comunicacao', 'informar', 'relatar', 'cep', 'patrocinador', 'documentar'])
    
    if tem_assistencia and tem_registro_notif:
        return 2, "Adequado: indicou o dever prioritário de assistência/segurança ao participante e o registro/notificação aos responsáveis (patrocinador/CEP)."
    elif tem_assistencia or tem_registro_notif:
        return 1, "Parcial: indicou apenas o cuidado assistencial OU apenas o dever de registro/notificação."
    else:
        return 0, "Incorreto ou genérico: não contemplou as condutas primordiais preconizadas em eventos adversos."

def avaliar_q8(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    aspectos = []
    if contem_termos(t, ['equipe', 'pessoas', 'profissionais', 'qualificad', 'treinad', 'habilitad', 'capacitad']):
        aspectos.append("Equipe qualificada/treinada")
    if contem_termos(t, ['estrutura', 'infraestrutura', 'espaco', 'equipamento', 'laboratorio', 'recursos', 'instalac', 'local', 'ambiente']):
        aspectos.append("Infraestrutura e recursos físicos")
    if contem_termos(t, ['recrutamento', 'participantes', 'pacientes', 'populacao', 'volume', 'amostra', 'cronograma']):
        aspectos.append("Capacidade de recrutamento e retenção")
    if contem_termos(t, ['qualidade', 'pops', 'procedimento', 'certificac', 'regulamentac', 'documentac', 'historico', 'sigilo']):
        aspectos.append("Sistemas de qualidade / conformidade ética")
        
    qtd = len(aspectos)
    if qtd >= 3:
        return 2, f"Adequado: identificou {qtd} aspectos determinantes ({', '.join(aspectos)})."
    elif qtd >= 1:
        return 1, f"Parcial: identificou {qtd} aspecto(s) ({', '.join(aspectos)}). O gabarito requer 3 aspectos."
    else:
        return 0, "Incorreto: não identificou os aspectos estruturantes para avaliar o preparo do centro."

def avaliar_q9(resp):
    t = normalizar_texto(resp)
    if not t or t in ['nao sei', 'nan']:
        return 0, "Sem resposta ou não soube responder."
        
    cuidados = []
    if contem_termos(t, ['tempo real', 'contemporaneo', 'momento', 'data e hora', 'completo', 'preciso', 'legivel', 'padronizado', 'padrao', 'assinad', 'id padrao', 'codigo', 'atividades realizadas']):
        cuidados.append("Registro contemporâneo, preciso e padronizado (ALCOA)")
    if contem_termos(t, ['armazenamento', 'seguro', 'seguranca', 'nuvem', 'controle de acesso', 'pastas', 'organizado', 'sigilo', 'confidencialidade', 'backup']):
        cuidados.append("Armazenamento seguro e controle de acesso")
    if contem_termos(t, ['controle de qualidade', 'auditoria', 'conferencia', 'revisao', 'rastreabilidade']):
        cuidados.append("Auditoria e controle de qualidade documental")
    if contem_termos(t, ['registro', 'registrar', 'documentacao']) and len(cuidados) == 0:
        cuidados.append("Registro de dados")
        
    qtd = len(cuidados)
    if qtd >= 2:
        return 2, f"Adequado: apontou cuidados consistentes ({', '.join(cuidados)})."
    elif qtd == 1:
        return 1, f"Parcial: apontou 1 cuidado ({', '.join(cuidados)}). Esperado: dois cuidados essenciais."
    else:
        return 0, "Incorreto: não apresentou cuidados essenciais relativos à integridade e rastreabilidade."

AVALIADORES = [avaliar_q1, avaliar_q2, avaliar_q3, avaliar_q4, avaliar_q5, avaliar_q6, avaliar_q7, avaliar_q8, avaliar_q9]

def processar_dataset(df_raw):
    """
    Processa um DataFrame do REDCap contendo respostas do inquérito.
    Retorna DataFrame enriquecido com scores, feedbacks e métricas.
    """
    colunas_questoes = df_raw.columns[4:13]
    col_slider = df_raw.columns[13] if len(df_raw.columns) > 13 else None
    
    registros = []
    for _, row in df_raw.iterrows():
        rec_id = row['Record ID']
        evento = row['Event Name']
        nome = str(row['NOME']).strip()
        sexo = str(row['SEXO']).strip()
        slider_val = row[col_slider] if col_slider and pd.notna(row[col_slider]) else np.nan
        
        dados_aluno = {
            "Record ID": rec_id,
            "Event Name": evento,
            "NOME": nome,
            "SEXO": sexo,
            "Autoavaliacao_Slider": slider_val
        }
        
        total_score = 0
        for i, func in enumerate(AVALIADORES):
            q_num = i + 1
            col_nome = colunas_questoes[i]
            resposta = row[col_nome]
            score, fb = func(resposta)
            total_score += score
            
            dados_aluno[f"Q{q_num}_Resposta"] = resposta if pd.notna(resposta) else ""
            dados_aluno[f"Q{q_num}_Score"] = score
            dados_aluno[f"Q{q_num}_Feedback"] = fb
            
        dados_aluno["Score_Total_18"] = total_score
        dados_aluno["Nota_10"] = round((total_score / 18.0) * 10.0, 2)
        dados_aluno["Percentual_Acerto"] = round((total_score / 18.0) * 100.0, 1)
        
        # Classificação de proficiência geral
        if total_score >= 14:
            classif = "Avançado / Alto Domínio"
        elif total_score >= 9:
            classif = "Intermediário / Conhecimento Parcial"
        else:
            classif = "Básico / Necessita Fortalecimento"
            
        dados_aluno["Classificacao_Geral"] = classif
        registros.append(dados_aluno)
        
    return pd.DataFrame(registros)
