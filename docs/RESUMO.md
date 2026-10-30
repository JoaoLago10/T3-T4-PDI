# Resumo da Etapa 1 — Classificação

Projeto em `/home/jglago/questoes-cursor/T3-T4-PDI`. Compara três classificadores pré-treinados no ImageNet em vídeo gravado e em webcam. Não houve treinamento novo.

## Modelos

Todos via `torchvision`, com `weights.transforms()` e `weights.meta["categories"]`. O dispositivo é CUDA se existir; os testes rodaram em CPU.

| Nome no código | Arquitetura | Pesos |
|---|---|---|
| `resnet50` | ResNet-50 | `IMAGENET1K_V2` |
| `efficientnet` | EfficientNet-B0 | `IMAGENET1K_V1` |
| `vit` | ViT-B/16 | `IMAGENET1K_V1` |

## Código

```text
classification/models.py                      # load_model(name)
classification/run_video.py                   # experimento A
classification/run_webcam.py                  # experimento B
classification/make_annotation_template.py    # frames + labels.csv vazio
classification/evaluate.py                    # métricas e gráficos
requirements.txt
README.md
```

- **Vídeo:** `--video`, `--model` (`resnet50|efficientnet|vit|all`), `--frame_step` (padrão 5), `--topk` (padrão 5). Warm-up de 10 inferências e `torch.cuda.synchronize()` na GPU. Gera `outputs/annotated_<modelo>.mp4`, `predictions_<modelo>.csv` e `metrics_<modelo>.json`.
- **Webcam:** `--model` e `--camera` (padrão 0). Mostra top-3 e FPS de média móvel. Teclas `1`, `2` e `3` trocam o modelo; `q` encerra e grava `outputs/webcam_<modelo>.csv` (FPS médio, latência média e p95).
- **Anotação:** extrai N frames (padrão 50) para `data/annotations/frames/` e cria `labels.csv` com colunas `frame_idx,true_label`. O nome precisa ser uma classe exata das 1000 do ImageNet.
- **Avaliação:** top-1, top-5, precisão, recall e F1 macro, matriz de confusão, parâmetros, latência, FPS, pico de GPU, `comparison.csv`, gráficos e `errors_<modelo>.csv`.

## Dados já usados

O arquivo `video-pdi-t3-t4.mp4` foi copiado de Downloads para `data/videos/teste.mp4`. O original foi preservado.

- Resolução: 464×832
- Duração: 28,7 s
- Quadros: 1648
- FPS do arquivo: 57,4
- Inferências por modelo: 330 (1 quadro a cada 5)

`data/annotations/labels.csv` tem 50 frames preenchidos e aprovados. Quando há vários objetos, o rótulo é o objeto principal.

- 25 `backpack`
- 13 `computer keyboard`
- 12 `water bottle`

## Ambientes

- WSL: `.venv` com PyTorch 2.5.1 CPU. A GPU não aparece no WSL e `/dev/video0` não existe.
- Windows, só para a webcam: `C:\Users\João Guilherme\.venvs\pdi-t3-t4`.

Webcam no PowerShell:

```powershell
cd \\wsl.localhost\Ubuntu\home\jglago\questoes-cursor\T3-T4-PDI
& "$env:USERPROFILE\.venvs\pdi-t3-t4\Scripts\python.exe" -m classification.run_webcam --model resnet50 --camera 0
```

## Resultados do vídeo

50 frames anotados, CPU. O FPS da tabela é só da inferência: `1000 / latência média`. Não há pico de memória de GPU.

| Modelo | Parâmetros | Top-1 | Top-5 | Latência | FPS | Erros | F1 macro |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 25,6 mi | 56% | 88% | 114,3 ms | 8,75 | 22 | 0,21 |
| EfficientNet-B0 | 5,3 mi | 26% | 74% | 39,2 ms | 25,52 | 37 | 0,07 |
| ViT-B/16 | 86,6 mi | 56% | 82% | 278,3 ms | 3,59 | 22 | 0,18 |

Top-1 / top-5 por classe:

| Classe | ResNet-50 | EfficientNet-B0 | ViT-B/16 |
|---|---|---|---|
| Mochila | 88% / 100% | 44% / 92% | 60% / 100% |
| Teclado | 0% / 69% | 0% / 46% | 38% / 54% |
| Garrafa | 50% / 83% | 17% / 67% | 67% / 75% |

A F1 macro é baixa porque cada classe errada inventada (`space bar`, `mailbag`, `safe`) pesa igual a uma classe verdadeira. A leitura principal é top-1 e top-5.

Erros que se repetem:

- Teclado pequeno na borda vira `space bar`.
- Mochila escura, sem alça visível, vira `mailbag` ou colete.
- Garrafa e mochila disputam o mesmo quadro.
- O ViT classificou uma garrafa como `safe` com 96% de confiança. Confiança alta não prova acerto.

## Resultados da webcam

Teste no Windows, CPU. As teclas `1`, `2`, `3` e `q` funcionaram. O FPS inclui captura, desenho e exibição, por isso é menor que o do vídeo.

| Modelo | Quadros | FPS | Latência média | p95 |
|---|---|---|---|---|
| ResNet-50 | 159 | 6,09 | 124,5 ms | 167,8 ms |
| EfficientNet-B0 | 410 | 15,17 | 42,2 ms | 50,8 ms |
| ViT-B/16 | 133 | 2,78 | 280,7 ms | 342,1 ms |

## Conclusão usada no relatório e na fala

Não há um vencedor único.

- **ResNet-50:** melhor equilíbrio. Melhor top-5 (88%) e custo intermediário.
- **ViT-B/16:** melhor leitura de teclado e garrafa, mas cerca de 7 vezes mais lento que o EfficientNet em CPU.
- **EfficientNet-B0:** 25,5 FPS no vídeo e 15,2 FPS na webcam, com top-1 de 26%.

A classificação olha o quadro inteiro. Como a cena tem 3 objetos, o passo natural seguinte é detecção. Não existe enunciado de Etapa 2 nesta conversa.

## Documentos

Pasta `docs/`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.tex` | Relatório completo em LaTeX |
| `apresentacao.tex` | 17 slides Beamer, 16:9, cartões coloridos |
| `roteiro-fala.tex` | Fala de cerca de 17 minutos, números em algarismos |
| `imagens/` | Gráficos e matrizes usados pelo LaTeX |
| `relatorio.zip` | Relatório + pasta `imagens/` |
| `apresentacao.zip` | Slides + pasta `imagens/` |
| `imagens.zip` | Só as figuras |

As figuras do LaTeX usam o caminho `imagens/`, não `../outputs/`. Não há `pdflatex` no WSL nem no Windows; a compilação foi orientada pelo Overleaf. Envie o zip e mantenha a pasta `imagens` ao lado do `.tex`.

Cores dos slides: azul para ResNet, verde para EfficientNet, roxo para ViT.

Divisão da fala:

- **João,** slides 1 a 6: capa, percurso, vídeo versus webcam, top-1 e top-5, mapa dos modelos e atalho da ResNet.
- **Wesley,** slides 7 a 9: EfficientNet, ViT e protocolo do teste.
- **André,** slides 10 a 16: placar, acerto por objeto, gráficos, erros, webcam, escolha e fechamento.
- Slide 17: referências, fica na tela durante as perguntas.

## O que não foi feito (Etapa 1)

- Commit git. No início, a pasta não era um repositório.
- PDF gerado localmente.

## Etapas 2–4 — pipeline completo (30/10/2026)

Mesmo vídeo, mesmos 50 rótulos. ImageNet → COCO: `backpack`, `water bottle`→`bottle`, `computer keyboard`→`keyboard`. Avaliação fraca de quadro (não há caixas/máscaras manuais). CPU. `--scene_only` ligado. Limiar 0,4.

### Detecção (`detection/`, `outputs/detection/`)

| Modelo | Recall | Mochila | Garrafa | Teclado | Latência | FPS |
|---|---|---|---|---|---|---|
| Faster R-CNN R50 FPN V2 | 52% | 32% | 83% | 62% | 6993 ms | 0,14 |
| Faster R-CNN MobileNet 320 | 32% | 0% | 75% | 54% | 141 ms | 7,08 |
| SSDLite MobileNetV3 | 18% | 24% | 25% | 0% | 103 ms | 9,70 |

Webcam: teclas 1/2/3, não testada no WSL. RetinaNet foi trocado pelo MobileNet 320 porque o ResNet-50 já toma ~42 min em CPU.

### Segmentação (`segmentation/`, `outputs/segmentation/`)

GrabCut e Otsu sobre as caixas do SSDLite. Recall 18% (herdado do detector). Otsu 12,0 FPS; GrabCut 8,3 FPS. Mask R-CNN está no código (`--model maskrcnn`); o vídeo completo não rodou em CPU porque custa o mesmo que o Faster R-CNN (~40 min).

Webcam: 1 Mask R-CNN, 2 GrabCut, 3 Otsu.

### Rastreamento (`tracking/`, `outputs/tracking/`)

Detector SSDLite + IoU, centróide e Kalman. Recall 18%. ~12 IDs únicos (fragmentação: o SSDLite some e o objeto volta com ID novo). Nenhuma trajetória misturou classes. ~12 FPS.

Webcam: 1 IoU, 2 centróide, 3 Kalman.

### Pipeline (`pipeline/`, `outputs/pipeline/`)

Uma passagem: GrabCut + Kalman. Vídeo `annotated_pipeline.mp4`. Cópia em `Downloads/pipeline_completo.mp4`. Webcam: `--segmenter threshold --tracker kalman`.
