income = float(input("Enter annual income: "))
is_senior = input("Are you a senior citizen (60+)? (yes/no): ").strip().lower() == "yes"

tax = 0.0

# Exemption limit based on age
exemption = 300000 if is_senior else 250000

if income <= exemption:
    tax = 0.0
elif income <= 500000:
    tax = (income - exemption) * 0.05
elif income <= 1000000:
    tax = (500000 - exemption) * 0.05 + (income - 500000) * 0.20
else:
    tax = (500000 - exemption) * 0.05 + (500000 * 0.20) + (income - 1000000) * 0.30

# Nested surcharge logic for high earners
if income > 5000000:
    if income <= 10000000:
        tax += tax * 0.10  # 10% surcharge
    else:
        tax += tax * 0.15  # 15% surcharge

print(f"Total Payable Tax: {tax:.2f}")
