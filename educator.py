import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

class FinancialAssistant:
    def __init__(self):
        self.financial_terms = [
            "stocks", "bonds", "mutual funds", "ETFs", "IRAs", "401k", " Roth IRA",
            "compound interest", "inflation", "budgeting", "debt", "credit score",
            "credit cards", "mortgage", "loans", "savings", "checking", "investing"
        ]

    def get_answer(self, query):
        doc = nlp(query)
        matcher = Matcher(nlp.vocab)

        for term in self.financial_terms:
            matcher.add("financial_term", [[{"LOWER": term}]])

        matches = matcher(doc)
        if matches:
            best_match_index = 0
            best_match_score = -1

            for match_id, start, end in matches:
                span = doc[start:end]
                match_score = span.text.lower() == term
                if match_score > best_match_score:
                    best_match_index = match_id
                    best_match_score = match_score

            matched_term = doc[matches[best_match_index][1]:matches[best_match_index][2]]
            return f"The term '{matched_term}' is a financial term related to {matched_term.text.lower()}."
        else:
            return "I'm not sure I understand the financial concept you're asking about."


assistant = FinancialAssistant()

while True:
    user_query = input("Enter your financial education topic or type 'quit' to exit: ")
    if user_query.lower() == "quit":
        break
    response = assistant.get_answer(user_query)
    print(response)