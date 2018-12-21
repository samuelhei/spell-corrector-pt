from spell_corrector_pt import SpellCorrector
import requests
import os

print('Downloading dictionary')
r = requests.get('https://github.com/pythonprobr/palavras/blob/master/palavras.txt?raw=true')
print('Dictionary downloaded')

dictionary =  r.content.decode('utf-8').lower().split("\n")
spell = SpellCorrector(dictionary)
print('Start training (it may take a while)')
spell.train()
print('Traning successfull')


if os.path.isdir('dictionary') == False:
    os.makedirs('dictionary')

print('Saving the model')
spell.dump_model('dictionary')
print('Finished :D')