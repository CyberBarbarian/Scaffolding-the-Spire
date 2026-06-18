"""Small smoke checks for the embedded headless runtime boundary."""


def test_headless_runtime_enters_combat_and_exports_actions(game):
    state = game.start(seed="bench-runtime-smoke")
    state = game.skip_neow(state)

    state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")

    assert state["decision"] == "combat_play"
    assert state["hand"]
    assert state["enemies"]
    assert any(card.get("can_play") for card in state["hand"])
