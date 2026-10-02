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

import numpy as np

from hermes.config import Settings


class WakeWordEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._on_trigger: Callable[[], None] | None = None
        self._last_trigger_time = 0.0
        self._cooldown_seconds = 3.0  # ignora triggers por 3s após acionamento
        self._model = None

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
    def _ensure_model(self):
        if self._model is None:
            from openwakeword.model import Model

            print(f"🔊 Carregando modelo de wake word: '{self.settings.wake_word_model}'...")
            self._model = Model(wakeword_models=[self.settings.wake_word_model])
            print("🎧 Ouvindo o wake word...")
        return self._model

    def feed(self, chunk: np.ndarray) -> None:
        """Recebe um chunk de áudio e verifica se o wake word foi falado.

        Chamado pelo loop principal (único consumidor do stream).
        """
        try:
            model = self._ensure_model()
            prediction = model.predict(chunk.flatten().astype(np.int16))
            score = float(prediction[self.settings.wake_word_model])

            now = time.time()
            if score > self.settings.wake_threshold and (now - self._last_trigger_time) > self._cooldown_seconds:
                self._last_trigger_time = now
                if self._on_trigger:
                    self._on_trigger()
        except Exception as exc:
            print(f"⚠️  Erro na detecção do wake word: {exc}")

    # ------------------------------------------------------------------
    def start(self, on_trigger: Callable[[], None]) -> None:
        """Inicia a detecção em uma thread separada (somente modo teclado).

        No modo wake word, o loop principal deve chamar `feed()` a cada chunk.
        """
        self._on_trigger = on_trigger
        if self.settings.keyboard_mode:
            thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            thread.start()
