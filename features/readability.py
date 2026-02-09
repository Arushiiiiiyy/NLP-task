# features/readability.py
import textstat

def fk_grade(text):
    return textstat.flesch_kincaid_grade(text)
