"""
core.py - Motor de audiodescricao automatica.
Contem as funcoes de processamento, sem interface.
Importado por app.py (Gradio) ou por scripts CLI.
"""

import os
import time
import json
import asyncio
import subprocess
import unicodedata
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import edge_tts
from google import genai

#Configuração
PASTA_SAIDA = Path("saida")
PASTA_SEGMENTOS = PASTA_SAIDA / "segmentos"

LIMIAR_SSIM = 0.65
INTERVALO_MIN = 4.0
INTERVALO_MAX = 12.0
INTERVALO_AMOSTRAGEM = 0.5

VOZ_TTS = "pt-BR-AntonioNeural"
VELOCIDADE_TTS = "+15%"

MODELO_GEMINI = "gemini-2.5-flash"
REDUCAO_DB_ORIGINAL = 15

FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_BIN = r"C:\ffmpeg\bin\ffprobe.exe"
YTDLP_BIN = "yt-dlp"  #Se yt-dlp nao estiver no PATH, troque por caminho absoluto

PASTA_SAIDA.mkdir(exist_ok=True)
PASTA_SEGMENTOS.mkdir(exist_ok=True)

#Gemini
def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Variavel GOOGLE_API_KEY nao definida")
    return genai.Client(api_key=api_key)

#Utilitários
def remover_acentos(txt):
    nfkd = unicodedata.normalize('NFKD', txt)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tem_audio(caminho_video):
    cmd = [
        FFPROBE_BIN, "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0",
        str(caminho_video)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return "audio" in r.stdout.lower()

#Prompt
def montar_prompt(historico):
    hist_txt = "\n".join(f"- {h}" for h in historico) if historico else "(nenhuma descricao anterior)"
    return f"""Voce e um audiodescritor profissional gerando descricao em tempo real para uma pessoa cega ou com baixa visao.

REGRAS INEGOCIAVEIS:
1. Maximo 15 palavras por descricao.
2. Use presente do indicativo.
3. Descreva o que VE, nunca interprete emocoes.
4. Priorize pessoas, acoes e mudancas de cenario.
5. Se houver texto na tela, leia-o literalmente.
6. Nao descreva o que o audio ja transmite.
7. Vocabulario simples e direto, sem metaforas.
8. Mantenha consistencia de nomenclatura com descricoes anteriores.

CONTEXTO DAS ULTIMAS DESCRICOES (mais recente por ultimo):
{hist_txt}

Descreva APENAS o que mudou ou e novo em relacao ao contexto acima.
Se nada relevante mudou, responda EXATAMENTE: [sem mudanca]"""

#Gemini
def analisar_frame_gemini(frame_bgr, historico, client, log_fn=print):
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if max(h, w) > 1024:
            escala = 1024 / max(h, w)
            frame_rgb = cv2.resize(frame_rgb, (int(w*escala), int(h*escala)))
        img = Image.fromarray(frame_rgb)
        caminho_tmp = PASTA_SAIDA / "_frame_tmp.jpg"
        img.save(caminho_tmp, quality=85)

        prompt = montar_prompt(historico)
        t0 = time.time()
        upload = client.files.upload(file=str(caminho_tmp))
        resp = client.models.generate_content(
            model=MODELO_GEMINI, contents=[upload, prompt]
        )
        dt = time.time() - t0
        texto = resp.text.strip()
        log_fn(f"  Gemini ({dt:.1f}s): {texto}")
        return texto
    except Exception as e:
        log_fn(f"  ERRO Gemini: {e}")
        return "[sem mudanca]"

#TTS
async def _gerar_tts(texto, caminho_mp3):
    com = edge_tts.Communicate(texto, VOZ_TTS, rate=VELOCIDADE_TTS)
    await com.save(caminho_mp3)

def gerar_tts(texto, caminho_mp3):
    asyncio.run(_gerar_tts(texto, caminho_mp3))

#Captura SRT
def capturar_srt(url_srt, caminho_saida, duracao_seg, log_fn=print):
    log_fn(f"Capturando {duracao_seg}s do stream SRT...")
    cmd = [
        FFMPEG_BIN, "-y", "-i", url_srt,
        "-t", str(duracao_seg), "-c", "copy", str(caminho_saida)
    ]
    subprocess.run(cmd, check=True)
    log_fn("Captura SRT concluida.")

#Captura Youtube
def capturar_youtube(url_youtube, caminho_saida, duracao_seg, log_fn=print):
    """
    Captura N segundos de uma live ou video do YouTube usando yt-dlp + ffmpeg.
    """
    log_fn(f"Obtendo URL direta do YouTube: {url_youtube}")
    # pega URL HLS direta
    cmd_url = [YTDLP_BIN, "-g", "-f", "best[ext=mp4]/best", url_youtube]
    r = subprocess.run(cmd_url, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp falhou: {r.stderr}")
    url_direta = r.stdout.strip().split("\n")[0]
    log_fn("URL obtida. Capturando stream...")

    cmd = [
        FFMPEG_BIN, "-y", "-i", url_direta,
        "-t", str(duracao_seg),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(caminho_saida)
    ]
    subprocess.run(cmd, check=True)
    log_fn("Captura YouTube concluida.")

#Loop Principal ======================
def processar_video(caminho_video, log_fn=print, progress_fn=None):
    log_fn(f"Abrindo video: {caminho_video}")
    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise RuntimeError("Nao consegui abrir o video")

    client = get_client()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps
    log_fn(f"FPS: {fps:.1f} | Duracao: {duracao:.1f}s | Frames: {total_frames}")

    historico = []
    descricoes = []
    ultimo_frame_pequeno = None
    ultimo_tempo_analise = -INTERVALO_MIN
    proximo_check = 0.0
    contador = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tempo_atual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if progress_fn and duracao > 0:
            progress_fn(min(tempo_atual / duracao, 1.0))

        if tempo_atual < proximo_check:
            continue
        proximo_check = tempo_atual + INTERVALO_AMOSTRAGEM

        pequeno = cv2.resize(frame, (320, 180))
        cinza = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)

        deve_analisar = False
        similaridade = 1.0

        if ultimo_frame_pequeno is None:
            deve_analisar = True
        else:
            similaridade = ssim(ultimo_frame_pequeno, cinza)
            tempo_desde = tempo_atual - ultimo_tempo_analise

            if tempo_desde < INTERVALO_MIN:
                deve_analisar = False
            elif similaridade < LIMIAR_SSIM:
                deve_analisar = True
            elif tempo_desde >= INTERVALO_MAX:
                deve_analisar = True

        if deve_analisar:
            log_fn(f"t={tempo_atual:5.1f}s sim={similaridade:.2f} -> analisar")
            texto = analisar_frame_gemini(frame, historico, client, log_fn)
            texto_limpo = remover_acentos(texto.strip())

            if "[sem mudanca]" not in texto_limpo.lower() and len(texto_limpo) > 3:
                contador += 1
                nome_audio = f"seg_{contador:03d}.mp3"
                caminho_audio = PASTA_SEGMENTOS / nome_audio
                try:
                    gerar_tts(texto_limpo, str(caminho_audio))
                    descricoes.append({
                        "id": contador,
                        "timestamp": round(tempo_atual, 2),
                        "texto": texto_limpo,
                        "arquivo_audio": nome_audio
                    })
                    historico.append(texto_limpo)
                    historico = historico[-3:]
                    log_fn(f"  -> seg {contador}: {texto_limpo}")
                except Exception as e:
                    log_fn(f"  ERRO TTS: {e}")

            ultimo_tempo_analise = tempo_atual
            ultimo_frame_pequeno = cinza
        else:
            if ultimo_frame_pequeno is None:
                ultimo_frame_pequeno = cinza

    cap.release()
    log_fn(f"Loop concluido. {len(descricoes)} descricoes.")
    return descricoes

#Mixagem
def mixar_final(caminho_video, descricoes, caminho_saida, log_fn=print):
    if not descricoes:
        log_fn("Nenhuma descricao gerada, copiando video original.")
        subprocess.run([FFMPEG_BIN, "-y", "-i", str(caminho_video),
                        "-c", "copy", str(caminho_saida)], check=True)
        return

    video_tem_audio = tem_audio(caminho_video)
    log_fn(f"Video tem audio: {video_tem_audio}")

    inputs = ["-i", str(caminho_video)]
    for d in descricoes:
        inputs += ["-i", str(PASTA_SEGMENTOS / d["arquivo_audio"])]

    partes = []
    labels_ad = []
    for i, d in enumerate(descricoes, start=1):
        delay_ms = int(d["timestamp"] * 1000)
        partes.append(f"[{i}:a]adelay={delay_ms}|{delay_ms},volume=2.0[ad{i}]")
        labels_ad.append(f"[ad{i}]")

    if len(descricoes) == 1:
        mix_ad = f"{labels_ad[0]}acopy[adall]"
    else:
        mix_ad = "".join(labels_ad) + f"amix=inputs={len(descricoes)}:duration=longest:normalize=0[adall]"
    partes.append(mix_ad)

    if video_tem_audio:
        reducao = 10 ** (-REDUCAO_DB_ORIGINAL / 20)
        partes.append(f"[0:a]volume={reducao:.3f}[orig]")
        partes.append("[orig][adall]amix=inputs=2:duration=first:normalize=0[final]")
    else:
        partes.append("[adall]apad[final]")

    filtro = ";".join(partes)

    cmd = [
    FFMPEG_BIN, "-y", *inputs,
    "-filter_complex", filtro,
    "-map", "0:v", "-map", "[final]",
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest",
    str(caminho_saida)
]
    
    log_fn("Mixando audio final...")
    subprocess.run(cmd, check=True)
    log_fn(f"Video final: {caminho_saida}")

#Automatizador
def pipeline_arquivo(caminho_video, nome_saida="final.mp4", log_fn=print, progress_fn=None):
    """Pipeline completo para arquivo local. Retorna caminho do video final."""
    #Limpa segmentos antigos
    for f in PASTA_SEGMENTOS.glob("*.mp3"):
        f.unlink()

    descricoes = processar_video(caminho_video, log_fn, progress_fn)
    caminho_final = PASTA_SAIDA / nome_saida
    mixar_final(caminho_video, descricoes, caminho_final, log_fn)

    #Salvar JSON também
    with open(PASTA_SAIDA / "descricoes.json", "w", encoding="utf-8") as f:
        json.dump(descricoes, f, ensure_ascii=False, indent=2)

    return str(caminho_final), descricoes

def pipeline_youtube(url_youtube, duracao_seg=60, nome_saida="final_yt.mp4", log_fn=print, progress_fn=None):
    capturado = PASTA_SAIDA / "captura_yt.mp4"
    capturar_youtube(url_youtube, capturado, duracao_seg, log_fn)
    return pipeline_arquivo(capturado, nome_saida, log_fn, progress_fn)

def pipeline_srt(url_srt, duracao_seg=60, nome_saida="final_srt.mp4", log_fn=print, progress_fn=None):
    capturado = PASTA_SAIDA / "captura_srt.mp4"
    capturar_srt(url_srt, capturado, duracao_seg, log_fn)
    return pipeline_arquivo(capturado, nome_saida, log_fn, progress_fn)
