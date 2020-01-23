import pytest
import src.profiler as pr
import src.drops as dp

class TestClass():
    def test_basic_init(self):
        seq = "agctgatgctgatgatgggc"
        prof = pr.Profiler(seq=seq, w=5, l =5, alias='XXX', family='YYY')
        assert prof.seq == seq
        assert prof.w == 5
        assert prof.l ==5
        assert prof.alias == 'XXX'
        assert prof.family == 'YYY'

    def test_invalid_seq(self):
        seq = 42
        with pytest.raises(Exception):
            pr.Profiler(seq=seq)

    def test_invalid_w(self):
        seq = "agctgatgctgatgatgggc"
        with pytest.raises(Exception):
            pr.Profiler(seq=seq, w=200)

    def test_invalid_l(self):
        seq = "agctgatgctgatgatgggc"
        with pytest.raises(Exception):
            pr.Profiler(seq=seq, l=200)

    def test_get_profiles(self):
        seq = "agctgatgctgatgatgggc"
        prof = pr.Profiler(seq=seq, w=5, l=5, alias='XXX', family='YYY')
        prof.get_profiles()
        assert prof.matrix is not None

    def test_generate_data(self):
        seq = "agctgatgctgatgatgggc"
        prof = pr.Profiler(seq=seq, w=5, l=5, alias='XXX', family='YYY')
        prof.generate_data()
        assert 'seq_id' in prof.matrix.columns and 'family' in prof.matrix.columns

    def test_drops(self):
        pass


