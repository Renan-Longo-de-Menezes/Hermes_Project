"""Detecção do wake word ("Hermes").

Dois modos:
1. Teclado  -> pressionar ENTER liga/desliga o modo escuta (para testes).
2. Wake word -> usa openWakeWord. OBS: o modelo padrão é "hey_jarvis";
   para usar literalmente a palavra "Hermes" é necessário treinar um
   modelo customizado (ex.: Porcupine) — deixamos isso para a Fase 1.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from hermes.config import Settings


class WakeWordEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._on_trigger: Callable[[], None] | None = None
        self._last_trigger_time = 0.0
        self._cooldown_seconds = 3.0  # ignora triggers por 3s após acionamento

    # ------------------------------------------------------------------
    # Modo teclado
    # ------------------------------------------------------------------
    def _keyboard_loop(self) -> None:
        print("️  Modo teclado: pressione ENTER para ligar/desligar o HERMES.")
        while True:
            input()  # aguarda ENTER
            if self._on_trigger:
                self._on_trigger()

    # ------------------------------------------------------------------
    # Modo wake word (openWakeWord)
    # ------------------------------------------------------------------
    def _wake_word_loop(self, stream) -> None:
        import numpy as np
        from openwakeword.model import Model

        print(f" Carregando modelo de wake word: '{self.settings.wake_word_model}'...")
        model = Model(wakeword_models=[self.settings.wake_word_model])
        print("🎧 Ouvindo o wake word...")

        while True:
            chunk, _ = stream.read(self.settings.chunk_samples)
            chunk = chunk.flatten().astype(np.int16)
            prediction = model.predict(chunk)
            score = float(prediction[self.settings.wake_word_model])
            
            # Cooldown: ignora triggers muito próximos
            now = time.time()
            if score > self.settings.wake_threshold and (now - self._last_trigger_time) > self._cooldown_seconds:
                self._last_trigger_time = now
                if self._on_trigger:
                    self._on_trigger()

    # ------------------------------------------------------------------
    def start(self, on_trigger: Callable[[], None]) -> None:
        """Inicia a detecção em uma thread separada."""
        self._on_trigger = on_trigger
        if self.settings.keyboard_mode:
            thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            thread.start()
        # Nota: o modo wake word agora recebe o stream via método `feed()`
        # (ver main.py)
