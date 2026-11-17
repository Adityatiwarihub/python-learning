print("--- Student Grade Calculator ---")

name = input("Enter student name: ")
marks = float(input("Enter total percentage (0 to 100): "))

if marks < 0 or marks > 100:
    print("Invalid input. Please enter marks between 0 and 100.")
elif marks >= 90:
    grade = "A+"
    status = "Distinction"
    print("Student:", name)
    print("Grade:", grade)
    print("Status:", status)
elif marks >= 75:
    grade = "A"
    status = "First Class"
    print("Student:", name)
    print("Grade:", grade)
    print("Status:", status)
elif marks >= 60:
    grade = "B"
    status = "Second Class"
    print("Student:", name)
    print("Grade:", grade)
    print("Status:", status)
elif marks >= 40:
    grade = "C"
    status = "Pass"
    print("Student:", name)
    print("Grade:", grade)
    print("Status:", status)
else:
    grade = "F"
    status = "Fail"
    print("Student:", name)
    print("Grade:", grade)
    print("Status:", status)
