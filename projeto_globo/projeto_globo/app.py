"""
app.py - Interface web Gradio do MVP de audiodescricao.
Rode com: python app.py
Acesse: http://localhost:7860
"""

import gradio as gr
import pandas as pd
import core

def processar_arquivo(arquivo_video, progress=gr.Progress()):
    if arquivo_video is None:
        return None, None, "Nenhum arquivo enviado."

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    def prog(pct):
        progress(pct, desc="Processando...")

    try:
        caminho_final, descricoes = core.pipeline_arquivo(
            arquivo_video, "final_upload.mp4", log_fn=log, progress_fn=prog
        )
        df = pd.DataFrame(descricoes) if descricoes else pd.DataFrame(
            columns=["id", "timestamp", "texto", "arquivo_audio"]
        )
        return caminho_final, df, "\n".join(logs[-30:])
    except Exception as e:
        return None, None, f"ERRO: {e}\n\n" + "\n".join(logs[-30:])


def processar_youtube(url_youtube, duracao, progress=gr.Progress()):
    if not url_youtube or not url_youtube.strip():
        return None, None, "URL vazia."

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    def prog(pct):
        progress(pct, desc="Processando...")

    try:
        progress(0.05, desc="Capturando do YouTube...")
        caminho_final, descricoes = core.pipeline_youtube(
            url_youtube.strip(), int(duracao), "final_yt.mp4",
            log_fn=log, progress_fn=prog
        )
        df = pd.DataFrame(descricoes) if descricoes else pd.DataFrame(
            columns=["id", "timestamp", "texto", "arquivo_audio"]
        )
        return caminho_final, df, "\n".join(logs[-30:])
    except Exception as e:
        return None, None, f"ERRO: {e}\n\n" + "\n".join(logs[-30:])

def processar_srt(url_srt, duracao, progress=gr.Progress()):
    if not url_srt or not url_srt.strip():
        return None, None, "URL SRT vazia."

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    def prog(pct):
        progress(pct, desc="Processando...")

    try:
        progress(0.05, desc="Capturando SRT...")
        caminho_final, descricoes = core.pipeline_srt(
            url_srt.strip(), int(duracao), "final_srt.mp4",
            log_fn=log, progress_fn=prog
        )
        df = pd.DataFrame(descricoes) if descricoes else pd.DataFrame(
            columns=["id", "timestamp", "texto", "arquivo_audio"]
        )
        return caminho_final, df, "\n".join(logs[-30:])
    except Exception as e:
        return None, None, f"ERRO: {e}\n\n" + "\n".join(logs[-30:])


with gr.Blocks(title="Audiodescritor Automatico MVP", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Audiodescritor Automatico — MVP")
    gr.Markdown(
        "Sistema de audiodescricao automatica em tempo quase-real para conteudo audiovisual. "
        "Combina deteccao inteligente de cena, IA multimodal (Gemini) e sintese de voz neural."
    )

    with gr.Tabs():
        # ABA 1 — Arquivo
        with gr.Tab("📁 Arquivo de vídeo"):
            with gr.Row():
                with gr.Column():
                    arq_input = gr.Video(label="Envie um vídeo MP4")
                    arq_btn = gr.Button("Gerar audiodescrição", variant="primary")
                with gr.Column():
                    arq_video_out = gr.Video(label="Vídeo final com audiodescrição")
            arq_tabela = gr.Dataframe(label="Descrições geradas", interactive=False)
            arq_log = gr.Textbox(label="Logs", lines=10, max_lines=10)
            arq_btn.click(processar_arquivo, inputs=[arq_input],
                          outputs=[arq_video_out, arq_tabela, arq_log])

        # ABA 2 — Youtube
        with gr.Tab("📺 Live do YouTube"):
            gr.Markdown("Cole a URL de uma live ou vídeo público do YouTube. Captura os primeiros N segundos.")
            with gr.Row():
                with gr.Column():
                    yt_url = gr.Textbox(label="URL do YouTube",
                                        placeholder="https://www.youtube.com/watch?v=...")
                    yt_duracao = gr.Slider(10, 180, value=60, step=10, label="Duração de captura (segundos)")
                    yt_btn = gr.Button("Capturar e processar", variant="primary")
                with gr.Column():
                    yt_video_out = gr.Video(label="Vídeo final com audiodescrição")
            yt_tabela = gr.Dataframe(label="Descrições geradas", interactive=False)
            yt_log = gr.Textbox(label="Logs", lines=10, max_lines=10)
            yt_btn.click(processar_youtube, inputs=[yt_url, yt_duracao],
                         outputs=[yt_video_out, yt_tabela, yt_log])

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
