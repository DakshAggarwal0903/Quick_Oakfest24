import random
from datetime import datetime, timedelta
import json
import os

def generate_transaction(transaction_count, current_salary, current_bonus):
    transaction_time = datetime(2024, 1, 1) + timedelta(minutes=random.randint(1, 365*24*60))
    
    # Increase salary and bonus amounts
    salary_increase = 20000.00
    bonus_increase = 1500.00
    
    if transaction_count % 100 == 0:
        transaction_details = "Bonus"
        current_bonus += bonus_increase
        transaction_amount = current_bonus
    elif transaction_count % 50 == 0:
        transaction_details = "Salary deposit"
        current_salary += salary_increase
        transaction_amount = current_salary
    else:
        # Realistic amounts for various expenses
        transaction_details = random.choice(["Groceries", "Gym membership", "Internet Bill", "Rent payment", "Clothing", "Monthly insurance payment", "Utilities", "Dining out", f"Purchase of Product {random.randint(1, 10)}"])
        transaction_amount = round(random.uniform(-500, -20), 2)
    
    transaction_id = random.randint(1000, 9999)
    return {
        "bank_balance_pre": None,
        "transaction_time": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_amount": transaction_amount,
        "transaction_details": transaction_details,
        "transaction_id": transaction_id
    }

def generate_test_data():
    transactions = []
    bank_balance = 42000.00  # Initial bank balance
    current_salary = 6228.00
    current_bonus = 1500.00
    
    for count in range(1, 501):  # Adjusted to 500 transactions
        transaction = generate_transaction(count, current_salary, current_bonus)
        transaction["bank_balance_pre"] = round(bank_balance, 2)
        bank_balance += transaction["transaction_amount"]
        transactions.append(transaction)
    
    return transactions

def save_json_file(data, file_path):
    filename = 'test_data.json'
    full_path = os.path.join(file_path, filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as file:
        json.dump(data, file, indent=2)

def main():
    person_data = {
        "name": "John Doe",
        "transactions": generate_test_data()
    }

    save_json_file(person_data, r'C:\Users\daksh\OneDrive\Desktop\Folders\Hackathons\Codefest\24')

if __name__ == "__main__":
    main()