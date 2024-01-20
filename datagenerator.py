import random
from datetime import datetime, timedelta
import json
import os

def generate_transaction(transaction_id, transaction_time, amount, category, description, account_balance_pre, savings_account_pre):
    return {
        "transaction_id": transaction_id,
        "transaction_time": transaction_time,
        "amount": amount,
        "category": category,
        "description": description,
        "account_balance_pre": account_balance_pre,
        "savings_account_pre": savings_account_pre
    }

def generate_savings_transaction(transaction_id, transaction_time, amount, description, savings_account_balance_pre):
    return {
        "transaction_id": transaction_id,
        "transaction_time": transaction_time,
        "amount": amount,
        "description": description,
        "savings_account_balance_pre": savings_account_balance_pre
    }

def generate_test_data():
    transactions = []
    savings_transactions = []
    account_balance = 42000.00  # Initial account balance
    savings_account_balance = 0.00  # Initial savings account balance
    monthly_income = 6228.00
    
    for count in range(1, 501):  # Generating 500 transactions
        transaction_time = datetime(2024, 1, 1) + timedelta(minutes=random.randint(1, 365*24*60))
        
        if count % 30 == 0:
            # Salary and bonus every 30 transactions
            transaction_amount = round(monthly_income + 500.00, 2)  # Bonus added
            category = "Salary and Bonus"
            description = "Salary and Bonus deposit"
            account_balance += transaction_amount
        else:
            # Other transactions represent expenses
            transaction_amount = round(random.uniform(-500, -20), 2)
            category = random.choice(["Rent", "Utilities", "Groceries", "Dining out", "Entertainment", "Clothing", "Others"])
            description = f"{category} expense"
            account_balance += transaction_amount
        
        transactions.append(
            generate_transaction(
                transaction_id=count,
                transaction_time=transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
                amount=transaction_amount,
                category=category,
                description=description,
                account_balance_pre=round(account_balance - transaction_amount, 2),
                savings_account_pre=round(savings_account_balance, 2)
            )
        )
        
        if count % 30 == 0:
            # Savings deposit every 30 transactions
            savings_increment = monthly_income * 0.25
            savings_account_balance += savings_increment
            savings_transactions.append(
                generate_savings_transaction(
                    transaction_id=5000 + count,
                    transaction_time=transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
                    amount=round(random.uniform(0.5, 2) * savings_increment, 2),
                    description="Savings deposit",
                    savings_account_balance_pre=round(savings_account_balance - savings_increment, 2)
                )
            )
    
    return {
        "user": {
            "name": "John Doe",
            "monthly_income": monthly_income,
            "savings_account_balance": round(savings_account_balance, 2)
        },
        "transactions": transactions,
        "savings_transactions": savings_transactions
    }

def save_json_file(data, file_path):
    filename = 'training_data.json'
    full_path = os.path.join(file_path, filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as file:
        json.dump(data, file, indent=2)

def main():
    training_data = generate_test_data()
    save_json_file(training_data, r"C:\Users\daksh\OneDrive\Desktop\Folders\Hackathons\Codefest'24'")

if __name__ == "__main__":
    main()
