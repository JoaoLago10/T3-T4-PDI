# Pipeline de visão — classificação, detecção, segmentação e rastreamento

O projeto compara modelos pré-treinados do `torchvision` em vídeo gravado e em
webcam. Não há treinamento novo. CUDA é usada automaticamente quando existe.

As quatro etapas usam o mesmo vídeo (`data/videos/teste.mp4`) e os 50 rótulos
de `data/annotations/labels.csv`. A classificação fala ImageNet; as etapas
seguintes falam COCO: `backpack`, `bottle` e `keyboard`.

## Estrutura

```text
data/
├── videos/
└── annotations/
    ├── labels.csv
    └── frames/
classification/     # Etapa 1
detection/          # Etapa 2
segmentation/       # Etapa 3
tracking/           # Etapa 4
pipeline/           # as três últimas em uma passagem
common/
outputs/
requirements.txt
```

## Instalação

Use Python 3.10 a 3.12. A partir da raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

As redes são baixadas na primeira execução. Confira o dispositivo:

```bash
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
```

## 1. Classificação (ImageNet)

Modelos: ResNet-50, EfficientNet-B0 e ViT-B/16.

```bash
python -m classification.make_annotation_template \
  --video data/videos/teste.mp4 --num_frames 50 --frame_step 5

python -m classification.run_video \
  --video data/videos/teste.mp4 --model all --frame_step 5 --topk 5

python -m classification.run_webcam --model resnet50 --camera 0

python -m classification.evaluate
```

Teclas da webcam: `1` ResNet-50, `2` EfficientNet-B0, `3` ViT-B/16, `q` sair.

## 2. Detecção (COCO)

Modelos: Faster R-CNN ResNet-50 FPN V2, Faster R-CNN MobileNetV3 320 e
SSDLite320 MobileNetV3.

```bash
python -m detection.run_video \
  --video data/videos/teste.mp4 --model all --frame_step 5

python -m detection.run_webcam --model ssdlite --camera 0

python -m detection.evaluate
```

Teclas da webcam: `1` Faster R-CNN, `2` MobileNet, `3` SSDLite, `q` sair.

A avaliação usa os 50 rótulos de quadro como verdade fraca: a classe ImageNet
é mapeada para COCO e o recall mede se essa classe apareceu em alguma caixa.
Não há caixas de referência manuais.

Saídas em `outputs/detection/`.

## 3. Segmentação

Métodos: Mask R-CNN ResNet-50 FPN V2, GrabCut sobre as caixas do SSDLite e
limiar de Otsu com morfologia sobre as mesmas caixas.

```bash
python -m segmentation.run_video \
  --video data/videos/teste.mp4 --model all --frame_step 5

python -m segmentation.run_webcam --model threshold --camera 0

python -m segmentation.evaluate
```

Teclas da webcam: `1` Mask R-CNN, `2` GrabCut, `3` Otsu, `q` sair.

Saídas em `outputs/segmentation/`.

## 4. Rastreamento

O detector padrão é o SSDLite. Rastreadores: IoU guloso, centróide e Kalman
de velocidade constante.

```bash
python -m tracking.run_video \
  --video data/videos/teste.mp4 --tracker all --frame_step 5

python -m tracking.run_webcam --tracker kalman --camera 0

python -m tracking.evaluate
```

Teclas da webcam: `1` IoU, `2` centróide, `3` Kalman, `q` sair.

Saídas em `outputs/tracking/`.

## Pipeline completo

Uma passagem: segmentação (caixas + máscaras) e rastreamento com ID estável.

```bash
python -m pipeline.run_video \
  --video data/videos/teste.mp4 \
  --segmenter maskrcnn \
  --tracker kalman \
  --frame_step 5

python -m pipeline.run_webcam --segmenter threshold --tracker kalman --camera 0
```

Saídas em `outputs/pipeline/`.

Em CPU, Mask R-CNN e Faster R-CNN são lentos. Para uma demo fluida da webcam,
prefira `--model ssdlite`, `--model threshold` e `--segmenter threshold`.

## Correspondência das classes

| Anotação ImageNet   | Classe COCO |
|---------------------|-------------|
| `backpack`          | `backpack`  |
| `water bottle`      | `bottle`    |
| `computer keyboard` | `keyboard`  |

O filtro `--scene_only` (ligado por padrão) mantém só essas três classes.

## Argumentos

```bash
python -m classification.run_video --help
python -m detection.run_video --help
python -m segmentation.run_video --help
python -m tracking.run_video --help
python -m pipeline.run_video --help
```
