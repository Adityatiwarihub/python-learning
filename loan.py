age = 28
annual_income = 650000  # in INR
credit_score = 760
has_existing_loan = False

if age >= 21 and age <= 60:
    if annual_income >= 500000 and credit_score >= 750:
        if not has_existing_loan:
            print("Loan Approved: Low Interest Rate")
        else:
            print("Loan Approved: Standard Interest Rate")
    elif annual_income >= 300000 and credit_score >= 700:
        print("Loan Approved: Co-signer Required")
    else:
        print("Loan Rejected: Income or Credit Score too low")
else:
    print("Loan Rejected: Age criteria not met")
