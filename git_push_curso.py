# -*- coding: utf-8 -*-
"""
Script de automação do Git Push para o repositório CURSOPC
Repositório: https://github.com/AlanKBF/CURSOPC.git
"""

import subprocess
import sys
import os

def rodar(cmd):
    print(f">> {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    return res.returncode

def main():
    diretorio = os.path.dirname(os.path.abspath(__file__))
    os.chdir(diretorio)
    
    msg = sys.argv[1] if len(sys.argv) > 1 else "Atualizacao automatica do projeto de pesquisa clinica"

    print("\n[INFO] Adicionando arquivos modificados...")
    rodar("git add .")
    
    print("\n[INFO] Realizando commit...")
    rodar(f'git commit -m "{msg}"')
    
    print("\n[INFO] Enviando para o GitHub...")
    code = rodar("git push origin main")
    
    if code == 0:
        print("\n" + "="*60)
        print("✅ PUSH REALIZADO COM SUCESSO NO GITHUB!")
        print("Repositório: https://github.com/AlanKBF/CURSOPC")
        print("Dashboard GitHub Pages: https://alankbf.github.io/CURSOPC/")
        print("="*60 + "\n")
    else:
        print("\n[AVISO] Tentando sincronização com origin/main...")
        rodar("git pull origin main --rebase")
        rodar("git push origin main")

if __name__ == "__main__":
    main()
