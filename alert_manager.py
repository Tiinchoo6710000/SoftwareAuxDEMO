import time

last_alert = 0
cooldown = 5

def trigger_alert(message):

    global last_alert

    now = time.time()

    if now - last_alert < cooldown:
        return

    print("\n============================")
    print("🚨 ALERTA DEL SISTEMA")
    print(message)
    print("============================\n")

    last_alert = now