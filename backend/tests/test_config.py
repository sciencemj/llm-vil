from llm_vil.config import load_llm_profiles


def test_village_layout(config):
    assert (config.width, config.height) == (40, 30)
    kinds = [o.kind for o in config.objects]
    assert kinds.count("farm_plot") == 10
    assert kinds.count("tree") >= 12
    assert kinds.count("berry_bush") == 6
    assert {
        "hearth", "notice_board", "trading_stall", "well",
        "bakery_oven", "forge_anvil", "kiln", "cauldron",
        "ore_node", "herb_patch", "shop_counter", "guild_counter",
        "monster", "gate_wood", "tavern", "guild_hall",
    } <= set(kinds)
    names = {v.name for v in config.villagers}
    assert names == {"민아", "준호", "서연", "도윤", "하늘", "태오", "강철", "바위", "리나", "현우"}
    wakes = sorted(v.wake_minute for v in config.villagers)
    sleeps = sorted(v.sleep_minute for v in config.villagers)
    assert 315 <= wakes[0] and wakes[-1] <= 420 and len(set(wakes)) >= 5
    assert 1245 <= sleeps[0] and sleeps[-1] <= 1350 and len(set(sleeps)) >= 5
    ids = {o.id for o in config.objects}
    for v in config.villagers:
        assert v.bed_id in ids and v.chest_id in ids and set(v.plots) <= ids
        assert config.area(v.home_area) is not None
        assert v.profession and v.skills
    for crop in config.crops.values():
        assert crop.seed_min >= 1  # harvest always yields >= 1 seed


def test_llm_profiles_have_no_secrets():
    profiles = load_llm_profiles()
    assert profiles["default_profile"] == "demo"
    for prof in profiles["profiles"].values():
        assert "api_key" not in prof
        assert prof["structured"] in {"native", "json_schema", "json_object", "prompt_only",
                                      "tool_call"}
