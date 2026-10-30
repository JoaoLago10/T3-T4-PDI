# Relatório do teste completo

Data: 30/10/2026. Ambiente: WSL2, Python 3.10.12, PyTorch 2.5.1+cpu, torchvision 0.20.1. Dispositivo: CPU (`torch.cuda.is_available()` falso). Não há `/dev/video0`; a abertura da câmera 0 falha de imediato.

Vídeo: `data/videos/teste.mp4`, 1648 quadros, 464×832, 57,4 fps, cerca de 28,7 s. Inferência a cada 5 quadros: 330 predições por modelo. Anotações: 50 quadros em `data/annotations/labels.csv` (25 `backpack`, 13 `computer keyboard`, 12 `water bottle`). Limiar 0,4 e `--scene_only` nas etapas COCO. O FPS das tabelas é só da inferência: `1000 / latência média`.

## O que foi executado

| Verificação | Resultado |
|---|---|
| `--help` dos 15 módulos de vídeo, webcam, avaliação e template | ok |
| Auditoria dos CSVs e MP4 já gerados | 330 linhas, índices 0, 5, …, 1645; vídeos com 1648 quadros em 464×832; os 50 rótulos casam com as predições |
| `classification.evaluate`, `detection.evaluate`, `tracking.evaluate` | exit 0; tabelas e gráficos regravados |
| `segmentation.run_video --model maskrcnn --frame_step 5` | exit 0 em 35,6 min (download dos pesos, aquecimento e 1648 quadros) |
| `segmentation.evaluate` com Mask R-CNN, GrabCut e Otsu | exit 0 |
| Webcam neste WSL | câmera inexistente; os CSV de 28/10 permanecem como registro anterior no Windows |

As inferências de classificação (28/10), detecção, GrabCut, Otsu, rastreamento e pipeline GrabCut+Kalman (30/10) foram conferidas e reavaliadas. A inferência nova desta sessão é o Mask R-CNN.

## 1. Classificação (ImageNet)

Arquivos: `outputs/comparison.csv`, `outputs/accuracy_vs_fps.png`, `outputs/accuracy_vs_latency.png`, `outputs/confusion_matrix_*.png`, `outputs/errors_*.csv`.

| Modelo | Parâmetros | Top-1 | Top-5 | F1 macro | Latência | FPS | Erros |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 25,6 mi | 56% | 88% | 0,21 | 114,3 ms | 8,75 | 22 |
| EfficientNet-B0 | 5,3 mi | 26% | 74% | 0,07 | 39,2 ms | 25,52 | 37 |
| ViT-B/16 | 86,6 mi | 56% | 82% | 0,18 | 278,3 ms | 3,59 | 22 |

Top-1 / top-5 por classe:

| Classe | ResNet-50 | EfficientNet-B0 | ViT-B/16 |
|---|---|---|---|
| Mochila (25) | 88% / 100% | 44% / 92% | 60% / 100% |
| Teclado (13) | 0% / 69% | 0% / 46% | 38% / 54% |
| Garrafa (12) | 50% / 83% | 17% / 67% | 67% / 75% |

Erros que mais se repetem: teclado vira `space bar` (ResNet, 10 vezes); EfficientNet concentra erros em `joystick` e `bulletproof vest`; o ViT troca mochila por `water bottle` (8 vezes).

Webcam no Windows, em 28/10 (captura, desenho e exibição entram no FPS):

| Modelo | Quadros | FPS | Latência | p95 |
|---|---:|---:|---:|---:|
| ResNet-50 | 159 | 6,09 | 124,5 ms | 167,8 ms |
| EfficientNet-B0 | 410 | 15,17 | 42,2 ms | 50,8 ms |
| ViT-B/16 | 133 | 2,78 | 280,7 ms | 342,1 ms |

## 2. Detecção (COCO)

Arquivos: `outputs/detection/comparison.csv`, `recall_vs_fps.png`, `recall_vs_latency.png`, `errors_*.csv`, `annotated_*.mp4`.

Recall: a classe ImageNet do quadro, mapeada para COCO, aparece em alguma caixa. Não há caixas manuais.

| Modelo | Recall | Mochila | Garrafa | Teclado | Precisão da classe | Caixas/quadro | Latência | FPS | Erros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN R50 FPN V2 | 52% | 32% | 83% | 62% | 30% | 1,27 | 6993 ms | 0,14 | 24 |
| Faster R-CNN MobileNet 320 | 32% | 0% | 75% | 54% | 28% | 0,75 | 141 ms | 7,08 | 34 |
| SSDLite MobileNetV3 | 18% | 24% | 25% | 0% | 18% | 0,23 | 103 ms | 9,70 | 41 |

A latência do Faster R-CNN é estável (mediana 6976 ms, máximo 8639 ms) nos 330 quadros. Quadros anotados sem nenhuma caixa da cena: 14, 21 e 36, na mesma ordem da tabela.

## 3. Segmentação

Arquivos: `outputs/segmentation/comparison.csv`, `recall_vs_fps.png`, `errors_*.csv`, `annotated_*.mp4`. O Mask R-CNN foi gerado nesta sessão: `annotated_maskrcnn.mp4` (1648 quadros), `predictions_maskrcnn.csv` (330 linhas), `metrics_maskrcnn.json`.

GrabCut e Otsu usam as caixas do SSDLite, então herdam o recall de 18%.

| Método | Recall | Mochila | Garrafa | Teclado | Caixas/quadro | Área média da máscara | Latência | FPS | Erros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mask R-CNN R50 FPN V2 | 28% | 4% | 83% | 23% | 1,08 | 13 869 px | 5761 ms | 0,17 | 36 |
| GrabCut + SSDLite | 18% | — | — | — | 0,23 | 13 598 px | 120 ms | 8,31 | 41 |
| Otsu + SSDLite | 18% | — | — | — | 0,23 | 14 357 px | 83 ms | 12,05 | 41 |

Mask R-CNN: 46,4 milhões de parâmetros, mediana 5706 ms, máximo 7743 ms, detecção em 62% dos quadros inferidos. A parede de execução do vídeo completo foi 35,6 min. O recall da mochila fica em 1/25.

## 4. Rastreamento

Detector SSDLite. Arquivos: `outputs/tracking/comparison.csv`, `ids_vs_length.png`, `errors_*.csv`.

| Rastreador | Recall | IDs únicos | Comprimento médio | Trilhas impuras | Trilhas/quadro | Latência | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| IoU guloso | 18% | 13 | 5,92 | 0 | 0,23 | 81,2 ms | 12,31 |
| Centróide | 18% | 12 | 6,42 | 0 | 0,23 | 79,0 ms | 12,67 |
| Kalman | 18% | 12 | 6,42 | 0 | 0,23 | 81,6 ms | 12,25 |

Há objeto em 77 dos 330 quadros. Nenhuma trilha misturou classes. Os cerca de 12 IDs mostram fragmentação: o detector some e o objeto volta com outro identificador.

## 5. Pipeline

`outputs/pipeline/metrics_pipeline.json`: GrabCut, Kalman, detector SSDLite, 330 quadros, 12 IDs, CPU, limiar 0,4. Latência média 119,5 ms. Vídeo: `outputs/pipeline/annotated_pipeline.mp4`.

## Leitura

No vídeo anotado, a ResNet-50 é o melhor equilíbrio da classificação (top-5 de 88% a 8,7 FPS). O Faster R-CNN é o detector com maior recall (52%) e custa cerca de 7 s por quadro em CPU. O Mask R-CNN segmenta de ponta a ponta com recall 28% e custo parecido (5,8 s). GrabCut e Otsu são rápidos e ficam limitados pelo SSDLite (recall 18%, teclado em 0%). Os três rastreadores preservam a classe e ficam em torno de 12 FPS; a diferença entre eles é de um ID.

A webcam ao vivo não foi repetida: neste WSL não existe câmera. Os CSV `outputs/webcam_*.csv` são do teste de classificação no Windows em 28/10.
