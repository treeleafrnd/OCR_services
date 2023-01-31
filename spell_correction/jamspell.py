import jamspell

def spell_corrector(text):
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('..\resources\en.bin')
    print(corrector.FixFragment(text))


