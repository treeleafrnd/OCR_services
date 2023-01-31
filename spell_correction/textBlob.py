from textblob import TextBlob

def spell_corrector(text):
    tb = TextBlob(text)
    corrected = tb.correct()
    print(corrected)