"""Detecção do wake word ("Hermes").

Dois modos:
1. Teclado  -> pressionar ENTER liga/desliga o modo escuta (para testes).
2. Wake word -> usa openWakeWord. OBS: o modelo padrão é "hey_jarvis";
   para usar literalmente a palavra "Hermes" é necessário treinar um
   modelo customizado (ex.: Porcupine) — deixamos isso para a Fase 1.
"""
from __future__ import annotations

import threading
from typing import Callable

from hermes.config import Settings


class WakeWordEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._on_trigger: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Modo teclado
    # ------------------------------------------------------------------
    def _keyboard_loop(self) -> None:
        print("⌨️  Modo teclado: pressione ENTER para ligar/desligar o HERMES.")
        while True:
            input()  # aguarda ENTER
            if self._on_trigger:
                self._on_trigger()

    # ------------------------------------------------------------------
    # Modo wake word (openWakeWord)
    # ------------------------------------------------------------------
    def _wake_word_loop(self) -> None:
        import numpy as np
        from openwakeword.model import Model

        print(f"🔉 Carregando modelo de wake word: '{self.settings.wake_word_model}'...")
        model = Model(wakeword_models=[self.settings.wake_word_model])

        import sounddevice as sd
        from hermes.audio.capture import AudioRecorder

        recorder = AudioRecorder(self.settings)
        print("🎧 Ouvindo o wake word...")

        with recorder.stream() as stream:
            while True:
                chunk, _ = stream.read(self.settings.chunk_samples)
                chunk = chunk.flatten().astype(np.int16)
                prediction = model.predict(chunk)
                score = float(prediction[self.settings.wake_word_model])
                if score > self.settings.wake_threshold and self._on_trigger:
                    self._on_trigger()

    # ------------------------------------------------------------------
    def start(self, on_trigger: Callable[[], None]) -> None:
        """Inicia a detecção em uma thread separada."""
        self._on_trigger = on_trigger
        if self.settings.keyboard_mode:
            thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        else:
            thread = threading.Thread(target=self._wake_word_loop, daemon=True)
        thread.start()
