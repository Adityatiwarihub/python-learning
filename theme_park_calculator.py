def calculate_ticket_price():
    print("=========================================")
    print("   WELCOME TO ADVENTURE THEME PARK! 🎢   ")
    print("=========================================")

    try:
        age = int(input("\nPlease enter the visitor's age: "))
    except ValueError:
        print("Invalid input! Please enter a valid whole number for age.")
        return

    # Validate age input
    if age < 0:
        print("Error: Age cannot be negative. Please try again.")
    elif age > 120:
        print("Error: Please enter a realistic age.")
        
    # Ticket pricing logic using if-elif-else
    elif age < 3:
        category = "Infant / Toddler"
        price = 0
        ride_access = "Kiddie Zone only (Must be accompanied by an adult)"
        print_receipt(age, category, price, ride_access)

    elif age <= 12:
        category = "Child"
        price = 25
        ride_access = "All Kids & Family Rides (Height restrictions may apply)"
        print_receipt(age, category, price, ride_access)

    elif age <= 17:
        category = "Teen"
        price = 40
        ride_access = "All Rides & Roller Coasters"
        print_receipt(age, category, price, ride_access)

    elif age <= 59:
        category = "Adult"
        price = 55
        ride_access = "Full Park Access"
        print_receipt(age, category, price, ride_access)

    else:
        category = "Senior Citizen"
        price = 30
        ride_access = "Full Park Access + Free Express Shuttle"
        print_receipt(age, category, price, ride_access)


def print_receipt(age, category, price, ride_access):
    print("\n--- TICKET SUMMARY ---")
    print(f"Visitor Age : {age} years")
    print(f"Category    : {category}")
    print(f"Ticket Price: ${price:.2f}")
    print(f"Access Level: {ride_access}")
    print("----------------------")
    print("Enjoy your day at the park! 🎡\n")


if __name__ == "__main__":
    calculate_ticket_price()
