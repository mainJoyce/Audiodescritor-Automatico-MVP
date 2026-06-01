# Audiodescritor Automático - MVP
---

## Projeto

O Audiodescritor Automático é um MVP desenvolvido para gerar audiodescrições de forma automatizada em conteúdos audiovisuais, como vídeos e trechos de lives.

A solução utiliza Inteligência Artificial multimodal para identificar mudanças de cena e gerar descrições objetivas, que são convertidas em áudio por meio de síntese de voz neural e incorporadas ao vídeo final.

---

## Funcionalidades

> Recebe vídeos enviados pelo usuário

> Analisa automaticamente mudanças de cena

> Gera descrições utilizando IA multimodal (Google Gemini)

> Converte descrições em áudio através de voz neural

> Sincroniza a audiodescrição com o vídeo

> Exporta uma nova versão acessível do conteúdo

> Possui interface web amigável desenvolvida com Gradio

---

## Suporte ao YouTube

O sistema permite o processamento de transmissões ao vivo disponíveis no YouTube.

Ao receber uma URL, a aplicação captura automaticamente um trecho de até 1 minuto da transmissão, analisa as cenas utilizando o Google Gemini e gera audiodescrições sincronizadas com o conteúdo capturado.

---

## Arquitetura da Solução

```text
Vídeo
  │
  ▼
Detecção de Mudanças de Cena
  │
  ▼
Google Gemini
(Geração das Descrições)
  │
  ▼
Edge TTS
(Síntese de Voz)
  │
  ▼
FFmpeg
(Mixagem de Áudio)
  │
  ▼
Vídeo Audiodescrito
```

---

## Tecnologias Utilizadas

| Tecnologia | Finalidade |
|------------|------------|
| Python | Linguagem principal |
| Gradio | Interface web |
| Google Gemini 2.5 Flash | Geração das descrições |
| OpenCV | Processamento de vídeo |
| Scikit-Image | Comparação estrutural entre frames |
| Edge TTS | Geração de voz |
| FFmpeg | Manipulação e mixagem de áudio/vídeo |
| Pillow | Processamento de imagens |

---

## Estrutura do Projeto

```text
projeto_globo/
│
├── app.py
├── core.py
├── requirements.txt
├── README.md
│
├── videos/
├── videos_prontos/
├── saida/
└── .gitignore
```

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

Entre na pasta:

```bash
cd projeto_globo
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente:

### Windows

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Configuração

Configure sua chave da API Gemini:

```bash
GOOGLE_API_KEY=sua_chave_aqui
```

Também é necessário possuir:

- FFmpeg instalado
- FFprobe instalado
- yt-dlp instalado (para vídeos do YouTube)

---

## Executando o Projeto

```bash
python app.py
```

Após iniciar a aplicação, acesse:

```text
http://localhost:7860
```

---

## Alunos

Ian Benia, Joyce Stefany, Samuel Souza e Thais Karol

---

## Uso do código

Projeto desenvolvido para fins acadêmicos.

---

