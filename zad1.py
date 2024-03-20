def total_euro(hours, payPerHour):
    return hours * payPerHour


hours = int(input("Unesite broj radnih sati:"))

payPerHour = int(input("Unesite iznos place za 1 sat:"))

totalPay = total_euro(hours, payPerHour)

print("Vasa ukupna placa iznosi: " + str(totalPay) + "€")
