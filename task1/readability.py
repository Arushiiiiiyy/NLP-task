# task1/readability.py
import textstat

def flesch_kincaid(text):
    """
    Flesch–Kincaid Grade Level.
    """
    return textstat.flesch_kincaid_grade(text)
