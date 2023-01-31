from spellchecker import SpellChecker
import re

def spell_corrector(text):
    w_string = ""
    spell = SpellChecker()
    words = re.split(' ',text)
    for word in words:
        corrected = str(spell.correction(word))
        if corrected == 'None':
            w_string = w_string + ' ' + word
        else:
            w_string = w_string + ' ' + corrected
    print(w_string)