"""Ponto de entrada do HERMES."""
import sys
from pathlib import Path

# Adiciona a pasta src/ ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import direto do módulo
from hermes.main import HermesApp

def main():
    try:
        app = HermesApp()
        app.run()
    except KeyboardInterrupt:
        print("\n HERMES encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
