from llm_vil.config import load_llm_profiles


def test_village_layout(config):
    assert (config.width, config.height) == (32, 24)
    kinds = [o.kind for o in config.objects]
    assert kinds.count("farm_plot") == 10
    assert kinds.count("tree") >= 12
    assert kinds.count("berry_bush") == 4
    assert {"hearth", "notice_board", "trading_stall", "well"} <= set(kinds)
    names = {v.name for v in config.villagers}
    assert names == {"민아", "준호", "서연", "도윤", "하늘"}
    wakes = sorted(v.wake_minute for v in config.villagers)
    sleeps = sorted(v.sleep_minute for v in config.villagers)
    assert 330 <= wakes[0] and wakes[-1] <= 390 and len(set(wakes)) == 5  # 05:30-06:30
    assert 1260 <= sleeps[0] and sleeps[-1] <= 1320 and len(set(sleeps)) == 5  # 21:00-22:00
    ids = {o.id for o in config.objects}
    for v in config.villagers:
        assert v.bed_id in ids and v.chest_id in ids and set(v.plots) <= ids
        assert config.area(v.home_area) is not None
    for crop in config.crops.values():
        assert crop.seed_min >= 1  # harvest always yields >= 1 seed


def test_llm_profiles_have_no_secrets():
    profiles = load_llm_profiles()
    assert profiles["default_profile"] == "demo"
    for prof in profiles["profiles"].values():
        assert "api_key" not in prof
        assert prof["structured"] in {"native", "json_schema", "json_object", "prompt_only",
                                      "tool_call"}
