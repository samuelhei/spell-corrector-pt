from spell_corrector_pt import SpellCorrector

spell = SpellCorrector()

try:
    spell.load_model('dictionary')
    print(spell.get_correct_word('abacat'))
    print(spell.get_correct_word('senoura'))
except:
    print('You need to train the model first, try python example-train.py')
