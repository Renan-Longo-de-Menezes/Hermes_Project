"""Ponto de entrada do HERMES."""
import sys
from pathlib import Path

# Adiciona a pasta src/ ao path para o Python encontrar o pacote hermes
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hermes.main import main

if __name__ == "__main__":
    main()
