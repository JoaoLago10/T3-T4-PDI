# Etapa 1 — Classificação de imagens

Este projeto compara ResNet-50, EfficientNet-B0 e ViT-B/16 pré-treinados no
ImageNet em vídeo gravado e em webcam. Cada modelo usa sua transformação
oficial do `torchvision`. CUDA é utilizada automaticamente quando disponível.

## Estrutura

```text
data/
├── videos/
└── annotations/
    ├── labels.csv
    └── frames/
classification/
├── models.py
├── run_video.py
├── run_webcam.py
├── evaluate.py
└── make_annotation_template.py
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

As redes são baixadas automaticamente na primeira execução. A versão de
PyTorch no `requirements.txt` funciona em CPU. Se precisar de uma variante
CUDA específica, instale `torch==2.5.1` e `torchvision==0.20.1` usando o índice
indicado pelo [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

Confira o dispositivo detectado:

```bash
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
```

## 1. Preparar as anotações

Coloque o vídeo em `data/videos/` e extraia 50 frames:

```bash
python -m classification.make_annotation_template \
  --video data/videos/meu_video.mp4 \
  --num_frames 50 \
  --frame_step 5
```

As imagens são gravadas em `data/annotations/frames/`. Preencha
`data/annotations/labels.csv` no formato:

```csv
frame_idx,true_label
0,coffee mug
125,laptop
```

Os modelos reconhecem somente as 1.000 classes do ImageNet. Portanto,
`true_label` deve ser exatamente um nome da lista oficial, incluindo espaços,
maiúsculas e pontuação. Para procurar o nome exato por uma parte do texto:

```bash
python -c "from torchvision.models import ResNet50_Weights as W; q='mug'; print([x for x in W.IMAGENET1K_V2.meta['categories'] if q.lower() in x.lower()])"
```

Use no processamento o mesmo `--frame_step` usado para criar o template.
Para recriar um CSV existente, acrescente `--overwrite`.

## 2. Experimento A — vídeo gravado

Executar os três modelos:

```bash
python -m classification.run_video \
  --video data/videos/meu_video.mp4 \
  --model all \
  --frame_step 5 \
  --topk 5
```

Executar apenas um modelo:

```bash
python -m classification.run_video \
  --video data/videos/meu_video.mp4 \
  --model efficientnet
```

Para cada modelo, o script cria em `outputs/`:

- `annotated_<modelo>.mp4`;
- `predictions_<modelo>.csv`;
- `metrics_<modelo>.json`, com parâmetros e pico de memória CUDA.

São realizadas dez inferências de aquecimento por padrão. A latência CUDA é
medida com sincronização explícita.

## 3. Experimento B — webcam

```bash
python -m classification.run_webcam --model resnet50 --camera 0
```

Durante a execução:

- `1`: ResNet-50;
- `2`: EfficientNet-B0;
- `3`: ViT-B/16;
- `q`: encerrar.

O top-3, as confianças e o FPS de média móvel aparecem na janela. Ao sair,
é criado `outputs/webcam_<modelo>.csv` para cada modelo utilizado.

## 4. Avaliação

Depois de preencher as anotações e gerar as predições:

```bash
python -m classification.evaluate \
  --labels data/annotations/labels.csv \
  --predictions_dir outputs \
  --output_dir outputs
```

Para avaliar apenas alguns modelos:

```bash
python -m classification.evaluate --models resnet50 vit
```

São produzidos:

- `comparison.csv`: top-1, top-5, precisão, recall, F1 macro e eficiência;
- `confusion_matrix_<modelo>.png`;
- `errors_<modelo>.csv`, para análise de oclusão, iluminação e escala;
- `accuracy_vs_fps.png`;
- `accuracy_vs_latency.png`.

O FPS comparativo é calculado como `1000 / latência média em ms` e representa
o throughput da inferência, sem incluir leitura, desenho ou gravação do vídeo.
A avaliação considera a interseção entre os `frame_idx` anotados e preditos e
avisa se alguma anotação não tiver predição.

## Argumentos disponíveis

Consulte todos os parâmetros e padrões de cada script com:

```bash
python -m classification.run_video --help
python -m classification.run_webcam --help
python -m classification.make_annotation_template --help
python -m classification.evaluate --help
```

Erros de vídeo inexistente, vídeo ilegível, codec inválido e webcam
indisponível são reportados com mensagens explícitas.
