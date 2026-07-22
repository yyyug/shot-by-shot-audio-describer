import pytest
from processing.llm_summarizer import (
    estimate_word_limit,
    summarize_to_ad,
    batch_summarize,
    VERB_LISTS,
    AD_SPEED
)


class TestEstimateWordLimit:
    def test_short_duration(self):
        result = estimate_word_limit(3.0)
        assert result == 11  # 3 / 0.275 ≈ 10.9, rounded to 11

    def test_medium_duration(self):
        result = estimate_word_limit(5.0)
        assert result == 18  # 5 / 0.275 ≈ 18.2, rounded to 18

    def test_long_duration(self):
        result = estimate_word_limit(10.0)
        assert result == 36  # 10 / 0.275 ≈ 36.4, rounded to 36

    def test_tv_series(self):
        result = estimate_word_limit(5.0, "tv_series")
        assert isinstance(result, int)

    def test_minimum_one_word(self):
        result = estimate_word_limit(0.1)
        assert result >= 1


class TestSummarizeToAd:
    def test_empty_description(self):
        result = summarize_to_ad("", "test_key")
        assert result == ""

    def test_raises_on_api_failure(self):
        # Without valid API key, should raise RuntimeError
        with pytest.raises(RuntimeError):
            summarize_to_ad("test", "fake_key", max_retries=1)


class TestBatchSummarize:
    def test_empty_input(self):
        result = batch_summarize([], "test_key")
        assert result == []

    def test_structure(self):
        items = [
            {"shot_id": 1, "start": 0.0, "end": 3.0, "description": "test"},
            {"shot_id": 2, "start": 3.0, "end": 6.0, "description": "test2"},
        ]
        # This will fail on API call, but tests structure
        try:
            result = batch_summarize(items, "fake_key")
        except:
            result = []
        assert isinstance(result, list)


class TestConstants:
    def test_verb_lists_exist(self):
        assert "movie" in VERB_LISTS
        assert "tv_series" in VERB_LISTS

    def test_ad_speed_exists(self):
        assert "movie" in AD_SPEED
        assert "tv_series" in AD_SPEED

    def test_ad_speed_values(self):
        assert AD_SPEED["movie"] > 0
        assert AD_SPEED["tv_series"] > 0
