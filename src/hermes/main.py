"""Orquestrador principal do HERMES — Fase 1C."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from hermes.audio.capture import AudioRecorder
from hermes.audio.diarizer import Diarizer, format_dialogue
from hermes.audio.wake_word import WakeWordEngine
from hermes.brain.classifier import Classifier
from hermes.brain.llm import Summarizer
from hermes.config import settings
from hermes.output.pdf_builder import PdfBuilder
from hermes.stt.transcriber import Transcriber


# Modos de análise disponíveis
MODES = {
    "1": ("resumo", "Resumo + Referências"),
    "2": ("pros_contras", "Prós e Contras"),
    "3": ("moral", "Análise Moral/Ética"),
    "auto": ("auto", "Automático (classifica e escolhe)"),
}


class HermesApp:
    def __init__(self):
        self.settings = settings
        self.recorder = AudioRecorder(self.settings)
        self.transcriber = Transcriber(self.settings)
        self.diarizer = Diarizer(self.settings)
        self.summarizer = Summarizer(self.settings)
        self.classifier = Classifier(self.settings)
        self.pdf_builder = PdfBuilder(self.settings)

        self.recording = False
        self.started_at = 0.0
        self.buffer: list[np.ndarray] = []
        self.current_mode = "1"  # default: resumo

    def on_trigger(self) -> None:
        if not self.recording:
            self._start()
        else:
            self._stop_and_process()

    def _start(self) -> None:
        self.recording = True
        self.buffer = []
        self.started_at = time.time()
        print("\n🎙️  HERMES ATIVADO — ouvindo a conversa... (acione novamente para parar)")

    def _stop_and_process(self) -> None:
        self.recording = False
        duration = time.time() - self.started_at
        print(f"\n⏹️  HERMES DESATIVADO — processando... ({len(self.buffer)} chunks capturados)")

        if not self.buffer:
            print("️  Nenhum áudio foi capturado.")
            return

        audio = np.concatenate(self.buffer)
        print(f"📊 Áudio total: {len(audio)} amostras ({duration:.1f}s)")

        wav_path = self.settings.temp_dir / "captura.wav"
        self.recorder.save(audio, wav_path, self.settings.sample_rate)

        try:
            # 1. Transcrição com timestamps
            print(f"📝 Transcrevendo ({wav_path.name})...")
            segments = self.transcriber.transcribe_segments(wav_path)

            # 2. Diariação (se habilitada)
            diarization = []
            if self.diarizer.enabled:
                print("🗣️  Identificando falantes...")
                try:
                    diarization = self.diarizer.diarize(wav_path)
                    print(f"    {len(diarization)} turnos de fala detectados.")
                except Exception as exc:
                    print(f"⚠️  Erro na diariação (seguindo sem rótulos): {exc}")
                    diarization = []

            # 3. Fusão transcrição + diarização
            merged = Diarizer.merge(segments, diarization)
            transcript_text = format_dialogue(merged)

            print("--- Transcrição (com falantes) ---")
            print(transcript_text or "(sem fala detectada)")
            print("-----------------------------------")

            if not transcript_text.strip():
                print("⚠️  Não foi possível detectar fala. Nenhum PDF foi gerado.")
                return

            # 4. Classificação de tema
            print("🏷️  Classificando tema...")
            tema = self.classifier.classify(transcript_text)
            print(f"    Tema detectado: {tema}")

            # 5. Análise via LLM (modo escolhido)
            mode_key = self.current_mode
            if mode_key == "auto":
                # Escolhe modo baseado no tema
                mode_key = self._auto_select_mode(tema)
            mode_id, mode_name = MODES[mode_key]
            print(f"📊 Modo de análise: {mode_name}")

            summary = self.summarizer.summarize(transcript_text, mode=mode_id)

            # 6. PDF bonito
            pdf_path = self.pdf_builder.build(
                summary=summary,
                transcript=transcript_text,
                duration_seconds=duration,
            )
            print(f"\n✅  PDF gerado em: {pdf_path}")

        except Exception as exc:
            print(f"\n  Erro durante o processamento: {exc}")

    def _auto_select_mode(self, tema: str) -> str:
        """Seleciona modo automaticamente baseado no tema."""
        # Debate → prós e contras
        if tema == "debate":
            return "2"
        # Aula, apresentação, notícia → resumo
        if tema in ("aula", "apresentação", "notícia", "entrevista"):
            return "1"
        # Reunião, conversa informal → análise moral (pode ter dilemas)
        if tema in ("reunião", "conversa informal"):
            return "3"
        return "1"  # default

    def run(self) -> None:
        banner = r"""
  _   _ _____ _____  __  __ _____ _____
 | | | | ____|  __ \|  \/  | ____/ ___|
 | |_| |  _| | |__) | |\/| |  _| \___ \
 |  _  | |___|  _  /| |  | | |___ ___) |
 |_| |_|_____|_| \_\|_|  |_|_____|____/

        Assistente de conversas — Fase 1C
        """
        print(banner)

        # Lista dispositivos de áudio
        print("🔊 Dispositivos de áudio disponíveis:")
        print(sd.query_devices())
        print(f"\n🎤 Entrada padrão: {sd.query_devices(kind='input')['name']}\n")

        # Menu de modos
        self._show_mode_menu()

        engine = WakeWordEngine(self.settings)
        engine.start(self.on_trigger)

        print("🎧 Abrindo stream de áudio...")
        try:
            with self.recorder.stream() as stream:
                print("✅ Stream aberto. Aguardando acionamento...\n")

                # Modo wake word: inicia detecção compartilhando o stream
                if not self.settings.keyboard_mode:
                    wake_thread = threading.Thread(
                        target=engine._wake_word_loop,
                        args=(stream,),
                        daemon=True,
                    )
                    wake_thread.start()

                try:
                    while True:
                        chunk, overflowed = stream.read(self.settings.chunk_samples)
                        if overflowed:
                            print("⚠️  Buffer overflow — áudio pode estar sendo perdido")
                        if self.recording:
                            self.buffer.append(chunk.flatten())
                            elapsed = time.time() - self.started_at
                            if elapsed > self.settings.max_recording_seconds:
                                print("⏱️  Limite de gravação atingido.")
                                self._stop_and_process()
                except KeyboardInterrupt:
                    print("\n👋  HERMES encerrado.")
        except Exception as e:
            print(f" Erro ao abrir stream: {e}")
            print("Tentando fallback com sd.rec()...")
            self._fallback_mode()

    def _show_mode_menu(self) -> None:
        """Exibe menu de seleção de modo."""
        print("📊 Modos de análise disponíveis:")
        for key, (_, name) in MODES.items():
            marker = " ← atual" if key == self.current_mode else ""
            print(f"   [{key}] {name}{marker}")
        print("\nDigite o número do modo (ou 'auto') e pressione ENTER:")

        try:
            choice = input("> ").strip().lower()
            if choice in MODES:
                self.current_mode = choice
                print(f"✅ Modo selecionado: {MODES[choice][1]}")
            else:
                print(f"⚠️  Opção inválida. Usando modo padrão: {MODES[self.current_mode][1]}")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Entrada cancelada. Usando modo padrão.")

    def _fallback_mode(self) -> None:
        """Modo fallback: grava X segundos direto com sd.rec()."""
        print("\n🎙️  Modo fallback: gravando 10 segundos direto...")
        duration = 10
        audio = sd.rec(
            int(duration * self.settings.sample_rate),
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype="int16",
        )
        sd.wait()
        print(f"✅ Gravado {duration}s. Processando...")

        wav_path = self.settings.temp_dir / "captura.wav"
        self.recorder.save(audio, wav_path, self.settings.sample_rate)

        segments = self.transcriber.transcribe_segments(wav_path)

        diarization = []
        if self.diarizer.enabled:
            try:
                diarization = self.diarizer.diarize(wav_path)
            except Exception as exc:
                print(f"⚠️  Erro na diariação: {exc}")

        merged = Diarizer.merge(segments, diarization)
        transcript_text = format_dialogue(merged)

        print("--- Transcrição ---")
        print(transcript_text or "(sem fala detectada)")
        print("-------------------")

        if transcript_text.strip():
            tema = self.classifier.classify(transcript_text)
            print(f"️  Tema: {tema}")

            mode_key = self.current_mode
            if mode_key == "auto":
                mode_key = self._auto_select_mode(tema)
            mode_id, mode_name = MODES[mode_key]
            print(f"📊 Modo: {mode_name}")

            summary = self.summarizer.summarize(transcript_text, mode=mode_id)
            pdf_path = self.pdf_builder.build(
                summary=summary,
                transcript=transcript_text,
                duration_seconds=duration,
            )
            print(f"\n✅  PDF gerado em: {pdf_path}")


def main() -> None:
    try:
        HermesApp().run()
    except KeyboardInterrupt:
        print("\n👋  HERMES encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
