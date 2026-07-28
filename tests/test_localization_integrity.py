from sts2_bench.localization_integrity import localization_integrity


def test_localization_integrity_accepts_normal_game_text():
    result = localization_integrity(
        (
            "Iron Wave deals 5 damage and gains 5 Block.",
            "Choose the next map room.",
        ),
        stderr="[INFO] Loaded 45 localization tables",
    )

    assert result == {
        "valid": True,
        "missing_directory_warning": False,
        "unresolved_tokens": [],
    }


def test_localization_integrity_rejects_raw_keys_and_missing_directory_warning():
    result = localization_integrity(
        (
            "CARD.IRON_WAVE.title",
            "EVENT.SOME_EVENT.description",
            "CHARACTER.IRONCLAD.name",
        ),
        stderr="[WARN] Localization dir not found: C:\\missing\\localization_eng",
    )

    assert result["valid"] is False
    assert result["missing_directory_warning"] is True
    assert result["unresolved_tokens"] == [
        "CARD.IRON_WAVE.title",
        "CHARACTER.IRONCLAD.name",
        "EVENT.SOME_EVENT.description",
    ]
