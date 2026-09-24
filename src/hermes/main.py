"""Orquestrador principal do HERMES (Fase 0) — versão robusta."""
from __future__ import annotations

import time 
from pathlib import Path

import numpy as np
import sounddevice as sd

from hermes.audio.capture import AudioRecorder
from hermes.audio.wake_word import WakeWordEngine
from hermes.brain.llm import Summarizer
from hermes.config import settings
from hermes.output.pdf_builder import PdfBuilder
from hermes.stt.transcriber import Transcriber 


class HermesApp:
    def __init__(self):
        self.settings = settings
        self.recorder = AudioRecorder(self.settings)
        self.transcriber = Transcriber(self.settings)
        self.summarizer = Summarizer(self.settings)
        self.pdf_builder = PdfBuilder(self.settings)

        self.recording = False
        self.started_at = 0.0
        self.buffer: list[np.ndarray] = []

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
        print(f"\n  HERMES DESATIVADO — processando... ({len(self.buffer)} chunks capturados)")

        if not self.buffer:
            print("⚠️  Nenhum áudio foi capturado.")
            return

        audio = np.concatenate(self.buffer)
        print(f"📊 Áudio total: {len(audio)} amostras ({len(audio)/self.settings.sample_rate:.1f}s)")

        wav_path = self.settings.temp_dir / "captura.wav"
        self.recorder.save(audio, wav_path, self.settings.sample_rate)

        try:
            print(f"📝 Transcrevendo ({wav_path.name})...")
            transcript = self.transcriber.transcribe(wav_path)
            print("--- Transcrição ---")
            print(transcript or "(sem fala detectada)")
            print("-------------------")

            if not transcript.strip():
                print("⚠️  Não foi possível detectar fala. Nenhum PDF foi gerado.")
                return

            summary = self.summarizer.summarize(transcript)
            pdf_path = self.pdf_builder.build(summary)
            print(f"\n✅  PDF gerado em: {pdf_path}")

        except Exception as exc:
            print(f"\n❌  Erro durante o processamento: {exc}")

    def run(self) -> None:
        banner = r"""
  _   _ _____ _____  __  __ _____ _____
 | | | | ____|  __ \|  \/  | ____/ ___|
 | |_| |  _| | |__) | |\/| |  _| \___ \
 |  _  | |___|  _  /| |  | | |___ ___) |
 |_| |_|_____|_| \_\|_|  |_|_____|____/

        Assistente de conversas — Fase 0
        """
        print(banner)

        # Lista dispositivos de áudio
        print("🔊 Dispositivos de áudio disponíveis:")
        print(sd.query_devices())
        print(f"\n🎤 Entrada padrão: {sd.query_devices(kind='input')['name']}\n")

        engine = WakeWordEngine(self.settings)
        engine.start(self.on_trigger)

        print("🎧 Abrindo stream de áudio...")
        try:
            with self.recorder.stream() as stream:
                print("✅ Stream aberto. Aguardando acionamento...\n")
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
            print(f"❌ Erro ao abrir stream: {e}")
            print("Tentando fallback com sd.rec()...")
            self._fallback_mode()

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

        transcript = self.transcriber.transcribe(wav_path)
        print("--- Transcrição ---")
        print(transcript or "(sem fala detectada)")
        print("-------------------")

        if transcript.strip():
            summary = self.summarizer.summarize(transcript)
            pdf_path = self.pdf_builder.build(summary)
            print(f"\n✅  PDF gerado em: {pdf_path}")


def main() -> None:
    HermesApp().run()


if __name__ == "__main__":
    main()
