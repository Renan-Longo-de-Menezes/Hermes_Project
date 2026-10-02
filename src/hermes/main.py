"""Orquestrador principal do HERMES — Fase 3."""
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
from hermes.dashboard.server import app, update_state, add_to_history
from hermes.output.pdf_builder import PdfBuilder
from hermes.stt.transcriber import Transcriber


MODES = {
    "1": ("resumo", "Resumo + Referências"),
    "2": ("pros_contras", "Prós e Contras"),
    "3": ("moral", "Análise Moral/Ética"),
    "4": ("todo", "Lista de Tarefas (TODO)"),
    "5": ("flashcards", "Flashcards (Anki)"),
    "6": ("ata", "Ata de Reunião"),
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
        self.current_mode = "1"

    def on_trigger(self) -> None:
        if not self.recording:
            self._start()
        else:
            self._stop_and_process()

    def _start(self) -> None:
        self.recording = True
        self.buffer = []
        self.started_at = time.time()
        update_state("recording")
        print("\n🎙️  HERMES ATIVADO — ouvindo a conversa... (acione novamente para parar)")

    def _stop_and_process(self) -> None:
        self.recording = False
        duration = time.time() - self.started_at
        update_state("processing")
        print(f"\n⏹️  HERMES DESATIVADO — processando... ({len(self.buffer)} chunks capturados)")

        if not self.buffer:
            print("⚠️  Nenhum áudio foi capturado.")
            update_state("idle")
            return

        audio = np.concatenate(self.buffer)
        print(f"📊 Áudio total: {len(audio)} amostras ({duration:.1f}s)")

        wav_path = self.settings.temp_dir / "captura.wav"
        self.recorder.save(audio, wav_path, self.settings.sample_rate)

        try:
            print(f"📝 Transcrevendo ({wav_path.name})...")
            segments = self.transcriber.transcribe_segments(wav_path)

            diarization = []
            if self.diarizer.enabled:
                print("🗣️  Identificando falantes...")
                try:
                    diarization = self.diarizer.diarize(wav_path)
                    print(f"    {len(diarization)} turnos de fala detectados.")
                except Exception as exc:
                    print(f"⚠️  Erro na diariação: {exc}")
                    diarization = []

            merged = Diarizer.merge(segments, diarization)
            transcript_text = format_dialogue(merged)

            print("--- Transcrição (com falantes) ---")
            print(transcript_text or "(sem fala detectada)")
            print("-----------------------------------")

            if not transcript_text.strip():
                print("⚠️  Não foi possível detectar fala.")
                update_state("idle")
                return

            print("🏷️  Classificando tema...")
            tema = self.classifier.classify(transcript_text)
            print(f"    Tema detectado: {tema}")

            mode_key = self.current_mode
            if mode_key == "auto":
                mode_key = self._auto_select_mode(tema)
            mode_id, mode_name = MODES[mode_key]
            print(f"📊 Modo de análise: {mode_name}")

            summary = self.summarizer.summarize(transcript_text, mode=mode_id)

            # Salva na memória
            if self.summarizer.memory.enabled:
                self.summarizer.memory.save(
                    transcript=transcript_text,
                    summary=summary,
                    tema=tema,
                    mode=mode_id,
                    duration=duration,
                )

            # Atualiza dashboard
            update_state("processing", transcript_text, summary)

            pdf_path = self.pdf_builder.build(
                summary=summary,
                transcript=transcript_text,
                duration_seconds=duration,
            )
            print(f"\n✅  PDF gerado em: {pdf_path}")

            # Salva no histórico do dashboard
            add_to_history(
                transcript=transcript_text,
                summary=summary,
                tema=tema,
                mode=mode_id,
                pdf_path=pdf_path,
                duration=duration,
            )

            # Exportação extra para Flashcards (CSV para Anki)
            if mode_id == "flashcards":
                csv_path = pdf_path.with_suffix(".csv")
                csv_content = self.summarizer.extractor.export_flashcards_csv(summary)
                csv_path.write_text(csv_content, encoding="utf-8")
                print(f"📇  Flashcards CSV exportado em: {csv_path}")
                print("    → Importe no Anki via: Arquivo > Importar")

            update_state("idle")

        except Exception as exc:
            print(f"\n❌  Erro durante o processamento: {exc}")
            update_state("idle")

    def _auto_select_mode(self, tema: str) -> str:
        if tema == "debate":
            return "2"
        if tema in ("aula", "apresentação", "notícia"):
            return "1"
        if tema == "reunião":
            return "6"
        if tema in ("entrevista", "conversa informal"):
            return "3"
        return "1"

    def run(self) -> None:
        banner = r"""
  _   _ _____ _____  __  __ _____ _____
 | | | | ____|  __ \|  \/  | ____/ ___|
 | |_| |  _| | |__) | |\/| |  _| \___ \
 |  _  | |___|  _  /| |  | | |___ ___) |
 |_| |_|_____|_| \_\|_|  |_|_____|____/

        Assistente de conversas — Fase 3
        """
        print(banner)

        print("🔊 Dispositivos de áudio disponíveis:")
        print(sd.query_devices())
        print(f"\n🎤 Entrada padrão: {sd.query_devices(kind='input')['name']}\n")

        # Status
        print("🧠 Fase 3 — Inteligência completa:")
        print(f"   Memória: {'✅' if self.summarizer.memory.enabled else '❌'} "
              f"({self.summarizer.memory.store.count if self.summarizer.memory.enabled else 0} registros)")
        print(f"   Referências: {'✅' if self.summarizer.references else '❌'}")
        print(f"   Extratores: ✅ (TODO, Flashcards, Ata)")
        print()

        # Inicia dashboard em thread separada
        def run_dashboard():
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")

        threading.Thread(target=run_dashboard, daemon=True).start()
        print("🌐 Dashboard disponível em: http://127.0.0.1:8765\n")

        self._show_mode_menu()

        engine = WakeWordEngine(self.settings)
        engine.start(self.on_trigger)

        print("🎧 Abrindo stream de áudio...")
        try:
            with self.recorder.stream() as stream:
                print("✅ Stream aberto. Aguardando acionamento...\n")

                # Loop ÚNICO de leitura do stream:
                # - no modo teclado, ENTER dispara on_trigger pela thread dedicada
                # - no modo wake word, cada chunk é alimentado no engine.feed()
                while True:
                    chunk, overflowed = stream.read(self.settings.chunk_samples)
                    if overflowed:
                        print("⚠️  Buffer overflow")

                    flat = chunk.flatten()

                    if not self.settings.keyboard_mode:
                        engine.feed(flat)

                    if self.recording:
                        self.buffer.append(flat)
                        elapsed = time.time() - self.started_at
                        if elapsed > self.settings.max_recording_seconds:
                            print("⏱️  Limite de gravação atingido.")
                            self._stop_and_process()

        except KeyboardInterrupt:
            print("\n👋  HERMES encerrado.")
        except Exception as e:
            print(f"❌ Erro ao abrir stream: {e}")
            print("Tentando fallback...")
            self._fallback_mode()

    def _show_mode_menu(self) -> None:
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
                print(f"⚠️  Opção inválida. Usando padrão: {MODES[self.current_mode][1]}")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Entrada cancelada.")

    def _fallback_mode(self) -> None:
        print("\n🎙️  Modo fallback: gravando 10 segundos...")
        duration = 10
        audio = sd.rec(
            int(duration * self.settings.sample_rate),
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype="int16",
        )
        sd.wait()
        print(f"✅ Gravado {duration}s.")

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
            mode_key = self.current_mode
            if mode_key == "auto":
                mode_key = self._auto_select_mode(tema)
            mode_id, _ = MODES[mode_key]
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
        print("\n👋  HERMES encerrado.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
