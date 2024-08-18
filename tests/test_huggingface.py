###################################################################################
# ocr_translate - a django app to perform OCR and translation of images.          #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate                                 #
###################################################################################
"""Test plugin."""
# pylint: disable=redefined-outer-name

from pathlib import Path

import pytest
from ocr_translate import models as m

from ocr_translate_hugging_face import plugin as hugginface
from ocr_translate_hugging_face.plugin.utils import EnvMixin, Loaders

# pytestmark = pytest.mark.django_db

@pytest.fixture(autouse=True)
def base(monkeypatch, tmpdir) -> Path:
    """Mock base classes."""
    tmp = str(tmpdir / 'base')
    monkeypatch.setenv('OCT_BASE_DIR', tmp)
    return Path(tmp)

@pytest.fixture()
def cache(monkeypatch, tmpdir) -> Path:
    """Mock base classes."""
    tmp = str(tmpdir / 'cache')
    monkeypatch.setenv('TRANSFORMERS_CACHE', tmp)
    return Path(tmp)

@pytest.fixture()
def  device(monkeypatch) -> str:
    """Mock base classes."""
    monkeypatch.setenv('DEVICE', 'cpu')
    return 'cpu'

@pytest.fixture()
def  device_cuda(monkeypatch) -> str:
    """Mock base classes."""
    monkeypatch.setenv('DEVICE', 'cuda')
    return 'cuda'

def test_env_none(monkeypatch):
    """Test that no env set causes ValueError."""
    monkeypatch.delenv('OCT_BASE_DIR', raising=False)
    with pytest.raises(ValueError):
        EnvMixin()

def test_env_transformers_cache(cache):
    """Test that the TRANSFORMERS_CACHE environment variable is set."""
    assert not cache.exists()
    mixin = EnvMixin()
    assert mixin.root == cache
    assert cache.exists()

def test_env_base_dir(base):
    """Test that the OCT_BASE_DIR environment variable is set."""
    assert not base.exists()
    mixin = EnvMixin()
    assert str(mixin.root).startswith(str(base))
    assert base.exists()

def test_env_transformers_cpu(device):
    """Test that the DEVICE environment variable is cpu."""
    mixin = EnvMixin()
    assert mixin.dev == device

def test_env_transformers_cuda(device_cuda):
    """Test that the DEVICE environment variable is cuda."""
    mixin = EnvMixin()
    assert mixin.dev == device_cuda

def test_load_hugginface_model_invalide_type():
    """Test high-level loading a huggingface model. Request unkown entity."""
    with pytest.raises(ValueError, match=r'^Unknown request: .*'):
        Loaders.load('test', ['invalid'], 'root')

def test_load_hugginface_model_return_none(monkeypatch):
    """Test high-level loading a huggingface model. Return None from load."""
    def mock_load(*args): # pylint: disable=unused-argument
        """Mocked load function."""
        return None
    monkeypatch.setattr(Loaders, '_load', mock_load)

    with pytest.raises(ValueError, match=r'^Could not load model: .*'):
        Loaders.load('test', ['model'], 'root')

@pytest.mark.parametrize('model_type', [
    'tokenizer',
    'ved_model',
    'model',
    'image_processor',
    'seq2seq'
])
def test_load_hugginface_model_success(monkeypatch, model_type):
    """Test high-level loading a huggingface model."""
    def mock_load(loader, *args): # pylint: disable=unused-argument
        """Mocked load function."""
        assert loader == Loaders.mapping[model_type]
        class App():
            """Mocked huggingface class with `to` method."""
            def to(self, x): # pylint: disable=invalid-name,unused-argument
                """Mocked method."""
                return None
        return App()
    monkeypatch.setattr(Loaders, '_load', mock_load)

    loaded = Loaders.load('test', [model_type], 'root')

    assert isinstance(loaded, dict)
    assert len(loaded) == 1
    assert model_type in loaded

####################################################################################
def test_load_from_storage_dir_fail(mock_loader, ved_model):
    """Test low-level loading a huggingface model from storage (missing file)."""
    # Load is supposed to test direcotry first and than fallback to cache
    # Exception should always be from not found in cache first
    with pytest.raises(FileNotFoundError, match='Not in cache'):
        ved_model.load()

def test_load_from_storage_dir_success(cache, mock_loader, ved_model):
    """Test low-level loading a huggingface model from storage (success)."""
    # monkeypatch.setattr(ved_model, 'root', tmp_dir)
    # Reload to make ENV effective
    # ved_model = hugginface.HugginfaceVEDModel.objects.get(name = ved_model.name)

    ptr = cache
    for pth in Path(ved_model.name).parts:
        new = ptr / pth
        new.mkdir(exist_ok=True)
        ptr = new
    ved_model.load()

def test_load_from_storage_cache_success(mock_loader, cache, ved_model):
    """Test low-level loading a huggingface model from storage (success)."""
    new = cache / ('models--' + ved_model.name.replace('/', '--'))
    new.mkdir(exist_ok=True)
    ved_model.load()

def test_unload_from_loaded_ved(monkeypatch, cache, ved_model):
    """Test unload box model with cpu."""
    monkeypatch.setattr(ved_model, 'model', '1')
    monkeypatch.setattr(ved_model, 'tokenizer', '1')

    ved_model.unload()
    assert ved_model.model is None
    assert ved_model.tokenizer is None

def test_unload_cpu(monkeypatch, cache, mock_called, ved_model):
    """Test unload box model with cpu."""
    monkeypatch.setattr(hugginface.ved.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(ved_model, 'dev', 'cpu')

    ved_model.unload()
    assert not hasattr(mock_called, 'called')

def test_unload_cuda(monkeypatch, cache, mock_called, ved_model):
    """Test unload box model with cuda."""
    monkeypatch.setattr(hugginface.ved.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(ved_model, 'dev', 'cuda')

    ved_model.unload()
    assert hasattr(mock_called, 'called')

# def test_pipeline_invalide_image(monkeypatch, hf_ved_model):
#     """Test ocr pipeline with invalid image."""
#     monkeypatch.setattr(hf_ved_model, 'model', '1')
#     monkeypatch.setattr(hf_ved_model, 'tokenizer', '1')
#     monkeypatch.setattr(hf_ved_model, 'image_processor', '1')
#     with pytest.raises(TypeError, match=r'^img should be PIL Image.*'):
#         hf_ved_model._ocr('invalid_image', 'ja') # pylint: disable=protected-access

def test_pipeline_notinit_ved(cache, ved_model):
    """Test tsl pipeline with not initialized model."""
    with pytest.raises(RuntimeError, match=r'^Model not loaded$'):
        ved_model._ocr('image') # pylint: disable=protected-access

def test_pipeline_hugginface(
        image_pillow, cache, mock_ocr_preprocessor, mock_ocr_tokenizer, mock_ocr_model, monkeypatch, ved_model):
    """Test ocr pipeline with hugginface model."""
    lang = 'ja'

    monkeypatch.setattr(ved_model, 'image_processor', mock_ocr_preprocessor(ved_model.name))
    monkeypatch.setattr(ved_model, 'tokenizer', mock_ocr_tokenizer(ved_model.name))
    monkeypatch.setattr(ved_model, 'model', mock_ocr_model(ved_model.name))

    res = ved_model._ocr(image_pillow, lang) # pylint: disable=protected-access

    assert res == 'abcde'

def test_pipeline_hugginface_cuda(
        image_pillow, mock_ocr_preprocessor, mock_ocr_tokenizer, mock_ocr_model, monkeypatch, ved_model):
    """Test ocr pipeline with hugginface model and cuda."""
    lang = 'ja'

    monkeypatch.setattr(ved_model, 'dev', 'cuda')
    monkeypatch.setattr(ved_model, 'image_processor', mock_ocr_preprocessor(ved_model.name))
    monkeypatch.setattr(ved_model, 'tokenizer', mock_ocr_tokenizer(ved_model.name))
    monkeypatch.setattr(ved_model, 'model', mock_ocr_model(ved_model.name))

    res = ved_model._ocr(image_pillow, lang) # pylint: disable=protected-access

    assert res == 'abcde'

####################################################################################
def test_get_mnt_wrong_options():
    """Test get_mnt with wrong options."""
    with pytest.raises(ValueError, match=r'^min_max_new_tokens must be less than max_max_new_tokens$'):
        hugginface.seq2seq.get_mnt(10, {'min_max_new_tokens': 20, 'max_max_new_tokens': 10})

def test_load_from_storage_dir_fail_s2s(mock_loader, s2s_model):
    """Test low-level loading a huggingface model from storage (missing file)."""
    # Load is supposed to test direcotry first and than fallnack to cache
    # Exception should always be from not found in cache first
    with pytest.raises(FileNotFoundError, match='Not in cache'):
        s2s_model.load()

def test_load_from_storage_dir_success_s2s(base, mock_loader, s2s_model):
    """Test low-level loading a huggingface model from storage (success)."""
    # Reload to make ENV effective
    # s2s_model = hugginface.HugginfaceSeq2SeqModel.objects.get(name = s2s_model.name)

    ptr = s2s_model.root
    for pth in Path(s2s_model.name).parts:
        new = ptr / pth
        ptr = new.mkdir(exist_ok=True)
        ptr = new
    s2s_model.load()

def test_load_from_storage_cache_success_s2s(base, mock_loader, s2s_model):
    """Test low-level loading a huggingface model from storage (success)."""
    pth = s2s_model.root / ('models--' + s2s_model.name.replace('/', '--'))
    pth.mkdir(exist_ok=True)
    s2s_model.load()

def test_unload_from_loaded_s2s(monkeypatch, s2s_model):
    """Test unload box model with cpu."""
    monkeypatch.setattr(s2s_model, 'model', '1')
    monkeypatch.setattr(s2s_model, 'tokenizer', '1')

    s2s_model.unload()
    assert s2s_model.model is None
    assert s2s_model.tokenizer is None

def test_unload_cpu_s2s(monkeypatch, mock_called, s2s_model):
    """Test unload box model with cpu."""
    monkeypatch.setattr(hugginface.seq2seq.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(s2s_model, 'dev', 'cpu')

    s2s_model.unload()
    assert not hasattr(mock_called, 'called')

def test_unload_cuda_s2s(monkeypatch, mock_called, s2s_model):
    """Test unload box model with cuda."""
    monkeypatch.setattr(hugginface.seq2seq.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(s2s_model, 'dev', 'cuda')

    s2s_model.unload()
    assert hasattr(mock_called, 'called')

def test_pipeline_notinit_s2s(s2s_model):
    """Test tsl pipeline with not initialized model."""
    with pytest.raises(RuntimeError, match=r'^Model not loaded$'):
        s2s_model._translate('test', 'ja', 'en') # pylint: disable=protected-access

def test_pipeline_wrong_type(monkeypatch, mock_tsl_model, mock_tsl_tokenizer, s2s_model):
    """Test tsl pipeline with wrong type."""
    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))
    with pytest.raises(TypeError):
        s2s_model._translate(1, 'ja', 'en') # pylint: disable=protected-access

def test_pipeline_no_tokens(monkeypatch, mock_tsl_tokenizer, s2s_model):
    """Test tsl pipeline with no tokens generated from pre_tokenize."""
    monkeypatch.setattr(s2s_model, 'model', '1')
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer('test/id'))

    res = s2s_model._translate([], 'ja', 'en') # pylint: disable=protected-access

    assert res == ''

def test_pipeline_m2m(monkeypatch, mock_tsl_tokenizer, mock_tsl_model, s2s_model):
    """Test tsl pipeline with m2m model."""
    monkeypatch.setattr(hugginface.seq2seq, 'M2M100Tokenizer', mock_tsl_tokenizer)
    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))

    s2s_model._translate(['1',], 'ja', 'en') # pylint: disable=protected-access

    assert s2s_model.tokenizer.called_get_lang_id is True


def test_pipeline(string, monkeypatch, mock_tsl_tokenizer, mock_tsl_model, mock_called, s2s_model):
    """Test tsl pipeline (also check that cache is not cleared in CPU mode)."""
    lang_src = 'ja'
    lang_dst = 'en'

    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))
    monkeypatch.setattr(hugginface.seq2seq.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(s2s_model, 'dev', 'cpu')

    res = s2s_model._translate([string,], lang_src, lang_dst) # pylint: disable=protected-access

    assert res == string
    assert s2s_model.tokenizer.model_id == s2s_model.name
    assert s2s_model.tokenizer.src_lang == lang_src

    assert not hasattr(mock_called, 'called')

def test_pipeline_clear_cache(monkeypatch, mock_tsl_tokenizer, mock_tsl_model, mock_called, s2s_model):
    """Test tsl pipeline with cuda should clear_cache."""
    lang_src = 'ja'
    lang_dst = 'en'

    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))
    monkeypatch.setattr(hugginface.seq2seq.torch.cuda, 'empty_cache', mock_called)
    monkeypatch.setattr(s2s_model, 'dev', 'cuda')

    s2s_model._translate(['test',], lang_src, lang_dst) # pylint: disable=protected-access

    assert hasattr(mock_called, 'called')



def test_pipeline_batch(batch_string, monkeypatch, mock_tsl_tokenizer, mock_tsl_model, s2s_model):
    """Test tsl pipeline with batched string."""
    lang_src = 'ja'
    lang_dst = 'en'

    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))

    batch_string = [[_] for _ in batch_string]
    res = s2s_model._translate(batch_string, lang_src, lang_dst) # pylint: disable=protected-access

    assert res == [_[0] for _ in batch_string]
    assert s2s_model.tokenizer.model_id == s2s_model.name
    assert s2s_model.tokenizer.src_lang == lang_src

@pytest.mark.parametrize(
    'options',
    [
        {},
        {'min_max_new_tokens': 30},
        {'max_max_new_tokens': 22},
        {'max_new_tokens': 15},
        {'max_new_tokens_ratio': 2}
    ],
    ids=[
        'default',
        'min_max_new_tokens',
        'max_max_new_tokens',
        'max_new_tokens',
        'max_new_tokens_ratio'
    ]
)
def test_pipeline_options(options, string, monkeypatch, mock_tsl_tokenizer, mock_tsl_model, s2s_model):
    """Test tsl pipeline with options."""
    lang_src = 'ja'
    lang_dst = 'en'

    monkeypatch.setattr(s2s_model, 'model', mock_tsl_model(s2s_model.name))
    monkeypatch.setattr(s2s_model, 'tokenizer', mock_tsl_tokenizer(s2s_model.name))

    min_max_new_tokens = options.get('min_max_new_tokens', 20)
    max_max_new_tokens = options.get('max_max_new_tokens', 512)
    ntok = string.replace('\n', ' ').count(' ') + 1

    string = m.TSLModel.pre_tokenize(string)
    if min_max_new_tokens > max_max_new_tokens:
        with pytest.raises(ValueError):
            s2s_model._translate(string, lang_src, lang_dst, options=options) # pylint: disable=protected-access
    else:
        s2s_model._translate(string, lang_src, lang_dst, options=options) # pylint: disable=protected-access

    mnt = hugginface.seq2seq.get_mnt(ntok, options)

    model = s2s_model.model

    assert model.options['max_new_tokens'] == mnt
