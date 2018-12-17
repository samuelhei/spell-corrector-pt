import pytest
from spell_corrector_pt import SpellCorrector, PathNotExistsException

@pytest.fixture
def spell():
    return SpellCorrector([
        'Abacate','Abacaxi','açaí',
        'acerola','amora', 'banana',
        'laranja'])

class TestClass(object):
    def test_dictionary(self, spell):
        assert len(spell.dictionary) == 7
        assert 'acai' in spell.clean_dictionary

    def test_inverted_text(self, spell):
        assert 'orvil' == spell.invert_text('livro')

    def test_non_existing_path(self, spell):
        path = 'nonexistingpath'
        with pytest.raises(PathNotExistsException) as excinfo:
            spell.load_model(path)
        assert path in str(excinfo.value)

        with pytest.raises(PathNotExistsException) as excinfo:
            spell.dump_model(path)
        assert path in str(excinfo.value)

    def test_dump_and_load(self, spell, tmp_path):
        spell.train()
        spell.dump_model(tmp_path)

        spell2 = SpellCorrector()
        spell2.load_model(tmp_path)

        assert bool(set(spell.dictionary).intersection(spell2.dictionary))
        assert bool(set(spell.phonetic).intersection(spell2.phonetic))
        assert bool(set(spell.comparation_dictionary).intersection(spell2.comparation_dictionary))
        assert bool(set(spell.vectorizer.vocabulary_).intersection(spell2.vectorizer.vocabulary_))

        assert 'banana' == spell2.get_correct_word('bannanana')
        assert 'Abacaxi' == spell2.get_correct_word('abacachi')
        assert 'laranja' == spell2.get_correct_word('laranga')
        assert 'acerola' == spell2.get_correct_word('acerloa')

    def test_get_probability(self, spell):
        spell.train()
        assert 'banana',1 == spell.get_correct_word('banana',return_probability=True)
        word, probability = spell.get_correct_word('banna',return_probability=True)
        assert word == 'banana'
        assert probability > 0

    def test_sub_phonetic(self, spell):
        assert 'abaksi' == spell.sub_phonetic('abacaxi')
        