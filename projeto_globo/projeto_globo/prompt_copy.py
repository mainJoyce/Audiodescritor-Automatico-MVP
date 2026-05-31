"""
MVP de Audiodescrição Automática em tempo quase-real
Autor: [seu nome]

Modos de uso:
    python prompt_copy.py arquivo videos/meu_video.mp4
    python prompt_copy.py srt srt://127.0.0.1:1234?mode=caller
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import edge_tts
from google import genai

#Configuração Centralizada
PASTA_SAIDA = Path("saida")
PASTA_SEGMENTOS = PASTA_SAIDA / "segmentos"
ARQUIVO_JSON = PASTA_SAIDA / "descricoes.json"

LIMIAR_SSIM = 0.65           #Abaixo disso = mudança de cena
INTERVALO_MIN = 4.0          #Segundos mínimos entre chamadas ao Gemini
INTERVALO_MAX = 12.0         #Segundos máximos sem análise
INTERVALO_AMOSTRAGEM = 0.5   #De quanto em quanto tempo verificar similaridade

VOZ_TTS = "pt-BR-AntonioNeural"
VELOCIDADE_TTS = "+15%"

MODELO_GEMINI = "gemini-2.5-flash"
DURACAO_CAPTURA_SRT = 100    # segundos a capturar do stream SRT
REDUCAO_DB_ORIGINAL = 6      # quanto abaixar o áudio original durante AD
FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"


#Iniciação
PASTA_SAIDA.mkdir(exist_ok=True)
PASTA_SEGMENTOS.mkdir(exist_ok=True)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERRO: defina a variavel GOOGLE_API_KEY")
    sys.exit(1)
client = genai.Client(api_key=api_key)


#Utilitários
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def remover_acentos(txt):
    nfkd = unicodedata.normalize('NFKD', txt)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


#Prompt Profissional
def montar_prompt(historico):
    hist_txt = "\n".join(f"- {h}" for h in historico) if historico else "(nenhuma descricao anterior)"
    return f"""Voce e um audiodescritor profissional gerando descricao em tempo real para uma pessoa cega ou com baixa visao.

REGRAS INEGOCIAVEIS:
1. Maximo 15 palavras por descricao.
2. Use presente do indicativo.
3. Descreva o que VE, nunca interprete emocoes.
4. Priorize pessoas, acoes e mudancas de cenario. Ignore detalhes decorativos.
5. Se houver texto na tela, leia-o literalmente.
6. Nao descreva o que o audio ja transmite.
7. Vocabulario simples e direto, sem metaforas.
8. Mantenha consistencia de nomenclatura com descricoes anteriores.

CONTEXTO DAS ULTIMAS DESCRICOES (mais recente por ultimo):
{hist_txt}

Descreva APENAS o que mudou ou e novo em relacao ao contexto acima.
Se nada relevante mudou, responda EXATAMENTE: [sem mudanca]"""


#Chama no Gemini
def analisar_frame_gemini(frame_bgr, historico):
    """Envia um frame para o Gemini e retorna a descricao."""
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        #Redimensiona pra max 1024px no lado maior

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
            model=MODELO_GEMINI,
            contents=[upload, prompt]
        )
        dt = time.time() - t0
        texto = resp.text.strip()
        log(f"  Gemini respondeu em {dt:.1f}s: {texto}")
        return texto
    except Exception as e:
        log(f"  ERRO Gemini: {e}")
        return "[sem mudanca]"


#TTS
async def gerar_tts(texto, caminho_mp3):
    com = edge_tts.Communicate(texto, VOZ_TTS, rate=VELOCIDADE_TTS)
    await com.save(caminho_mp3)


#Captura de Live SRT
def capturar_srt(url_srt, caminho_saida):
    log(f"Capturando {DURACAO_CAPTURA_SRT}s do stream SRT...")
    cmd = [
        FFMPEG_BIN, "-y", "-i", url_srt,
        "-t", str(DURACAO_CAPTURA_SRT),
        "-c", "copy", str(caminho_saida)
    ]
    subprocess.run(cmd, check=True)
    log("Captura SRT concluida.")


#Loop Principal - Detectar Cena
def processar_video(caminho_video):
    log(f"Abrindo video: {caminho_video}")
    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        log("ERRO: nao consegui abrir o video")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps
    log(f"FPS: {fps:.1f} | Duracao: {duracao:.1f}s | Frames: {total_frames}")

    historico = []
    descricoes = []
    ultimo_frame_pequeno = None
    ultimo_tempo_analise = -INTERVALO_MIN  # permite analisar logo no inicio

    proximo_check = 0.0
    contador = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tempo_atual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if tempo_atual < proximo_check:
            continue
        proximo_check = tempo_atual + INTERVALO_AMOSTRAGEM

        #Versão Pequena pra Comparação
        pequeno = cv2.resize(frame, (320, 180))
        cinza = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)

        deve_analisar = False
        similaridade = 1.0

        if ultimo_frame_pequeno is None:
            deve_analisar = True
            motivo = "primeiro frame"
        else:
            similaridade = ssim(ultimo_frame_pequeno, cinza)
            tempo_desde_ultima = tempo_atual - ultimo_tempo_analise

            if tempo_desde_ultima < INTERVALO_MIN:
                deve_analisar = False
                motivo = f"intervalo min nao atingido ({tempo_desde_ultima:.1f}s)"
            elif similaridade < LIMIAR_SSIM:
                deve_analisar = True
                motivo = f"mudanca de cena (sim={similaridade:.2f})"
            elif tempo_desde_ultima >= INTERVALO_MAX:
                deve_analisar = True
                motivo = f"intervalo max atingido ({tempo_desde_ultima:.1f}s)"
            else:
                deve_analisar = False
                motivo = f"sem mudanca relevante (sim={similaridade:.2f})"

        log(f"t={tempo_atual:5.1f}s sim={similaridade:.2f} -> {motivo}")

        if deve_analisar:
            texto = analisar_frame_gemini(frame, historico)
            texto_limpo = remover_acentos(texto.strip())

            if "[sem mudanca]" not in texto_limpo.lower() and len(texto_limpo) > 3:
                contador += 1
                nome_audio = f"seg_{contador:03d}.mp3"
                caminho_audio = PASTA_SEGMENTOS / nome_audio
                try:
                    asyncio.run(gerar_tts(texto_limpo, str(caminho_audio)))
                    descricoes.append({
                        "id": contador,
                        "timestamp": round(tempo_atual, 2),
                        "texto": texto_limpo,
                        "arquivo_audio": nome_audio
                    })
                    historico.append(texto_limpo)
                    historico = historico[-3:]  # mantem so as 3 ultimas
                    log(f"  -> segmento {contador} gerado: {texto_limpo}")
                except Exception as e:
                    log(f"  ERRO TTS: {e}")
            else:
                log("  -> descartado ([sem mudanca] ou vazio)")

            ultimo_tempo_analise = tempo_atual
            ultimo_frame_pequeno = cinza
        else:
            if ultimo_frame_pequeno is None:
                ultimo_frame_pequeno = cinza

    cap.release()
    log(f"Loop concluido. {len(descricoes)} descricoes geradas.")
    return descricoes

#Mixagem Final com Ducking
def mixar_final(caminho_video, descricoes, caminho_saida):
    if not descricoes:
        log("Nenhuma descricao para mixar.")
        return

    log("Detectando se o video original tem audio...")
    cmd_probe = [
        FFMPEG_BIN.replace("ffmpeg.exe", "ffprobe.exe"),
        "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", str(caminho_video)
    ]
    r = subprocess.run(cmd_probe, capture_output=True, text=True)
    video_tem_audio = "audio" in r.stdout.lower()
    log(f"Video tem audio: {video_tem_audio}")

    log("Construindo comando ffmpeg de mixagem...")
    inputs = ["-i", str(caminho_video)]
    for d in descricoes:
        inputs += ["-i", str(PASTA_SEGMENTOS / d["arquivo_audio"])]

    partes = []
    labels_ad = []
    for i, d in enumerate(descricoes, start=1):
        delay_ms = int(d["timestamp"] * 1000)
        partes.append(f"[{i}:a]adelay={delay_ms}|{delay_ms},volume=1.5[ad{i}]")
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
        FFMPEG_BIN, "-y",
        *inputs,
        "-filter_complex", filtro,
        "-map", "0:v",
        "-map", "[final]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(caminho_saida)
    ]

    log("Executando ffmpeg (mixagem final)...")
    subprocess.run(cmd, check=True)
    log(f"Video final gerado: {caminho_saida}")

#Orquestrador
def main():
    if len(sys.argv) < 3:
        print("Uso:")
        print("  python mvp.py arquivo <caminho_do_video>")
        print("  python mvp.py srt <url_srt>")
        sys.exit(1)

    modo = sys.argv[1]
    fonte = sys.argv[2]

    if modo == "srt":
        caminho_video = PASTA_SAIDA / "captura_srt.mp4"
        capturar_srt(fonte, caminho_video)
        nome_base = "srt_demo"
    elif modo == "arquivo":
        caminho_video = Path(fonte)
        if not caminho_video.exists():
            print(f"Arquivo nao encontrado: {fonte}")
            sys.exit(1)
        nome_base = caminho_video.stem
    else:
        print("Modo invalido. Use 'arquivo' ou 'srt'.")
        sys.exit(1)

    descricoes = processar_video(caminho_video)

    #Salva no JSON
    with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(descricoes, f, ensure_ascii=False, indent=2)
    log(f"JSON salvo: {ARQUIVO_JSON}")

    #Mixa o final
    caminho_final = PASTA_SAIDA / f"final_{nome_base}.mp4"
    mixar_final(caminho_video, descricoes, caminho_final)

    log("=== PROCESSO CONCLUIDO ===")
    log(f"Video final: {caminho_final}")
    log(f"Segmentos: {PASTA_SEGMENTOS}")
    log(f"JSON: {ARQUIVO_JSON}")

if __name__ == "__main__":
    main()
