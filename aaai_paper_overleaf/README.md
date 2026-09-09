# Artigo AAAI — análise por camada (Information, Behavior, and Format)

Pasta pronta para upload no Overleaf. Gerada a partir do repositório `~/msc`
(dissertação PortParser / Porttinari).

## Conteúdo

```
main.tex                              artigo completo (formato AAAI 2026)
layerwise_aaai.bib                    referências BibTeX
latex_tables/layerwise_aaai_tables.tex  7 tabelas (geradas programaticamente)
imagens/layerwise_fig_*.pdf           5 figuras vetoriais (fontes TrueType)
```

## Como compilar no Overleaf

1. **Baixe o AAAI 2026 Author Kit** em https://aaai.org/authorkit26/
   e copie para a RAIZ do projeto Overleaf:
   - `aaai2026.sty`
   - `aaai2026.bst`
2. Faça upload desta pasta inteira (ou do `aaai_paper_overleaf.zip`).
3. Compilador: **pdfLaTeX** | documento principal: `main.tex`.
4. A submissão usa `\usepackage[submission]{aaai2026}` (anônimo). Na versão
   camera-ready, troque para `[final]` e preencha `\author`/`\affiliations`.

## Pendências antes de submeter

- [ ] Substituir a entrada `porttinari2023` em `layerwise_aaai.bib` pela
      referência bibliográfica oficial do corpus Porttinari.
- [ ] Preencher autores/afiliações na versão final.
- [ ] Conferir limite de páginas do ano da submissão (AAAI: 7+2; ajustar
      figuras/tabelas para o apêndice se necessário).

## Regeneração dos assets (no repositório ~/msc)

Tabelas e figuras são geradas por scripts — **não editar à mão**:

```
layerwise_comparison_aaai.py     figuras 1–5 + todas as tabelas + significância
layerwise_probes.py              probes por camada (5 seeds, 5 encoders)
layerwise_mbert_analysis.py      análises mBERT (linear + biaffine)
layerwise_large_analysis.py      BERTimbau-large (24 camadas)
layerwise_independent_runs.py    replicação em runs independentes
layerwise_frozen_heads.py        cabeças re-treinadas no encoder congelado
```

Após regenerar, copie os arquivos atualizados para esta pasta
(`latex_tables/`, `imagens/`) — ou rode novamente o comando de montagem
documentado no histórico do projeto.

O texto-fonte da discussão/abstract também vive em
`~/msc/layerwise_aaai_discussion.tex` (manter sincronizado com `main.tex`).
