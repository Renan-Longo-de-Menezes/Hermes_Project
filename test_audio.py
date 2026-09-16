import sounddevice as sd

print("Dispositivos de áudio disponíveis:\n")
print(sd.query_devices()) 
print("\nDispositivo padrão de entrada:")
print(sd.query_devices(kind='input'))
