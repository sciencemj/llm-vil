import pytest

from llm_vil.agent.conversation import ConversationError, ConversationSession
from llm_vil.schemas.agent import ConversationTurn


def turn(s, tick, end=False, gift="none", valence=0.3):
    sp = s.next_speaker
    return ConversationTurn(speaker_id=sp, listener_id=s.partner_of(sp), tick=tick,
                            utterance="안녕?", end_conversation=end, gift_item=gift,
                            valence=valence)


def session(**kw):
    return ConversationSession(id="c1", initiator_id="mina", target_id="junho", topic="농사",
                               start_tick=0, **kw)


def test_full_lifecycle_max_turns():
    s = session(max_turns=6)
    s.accept(0)
    for i in range(6):
        s.add_turn(turn(s, i * 2, gift="berry" if i == 1 else "none"), 2)
    assert s.state == "ended" and s.end_reason == "max_turns"
    out = s.outcome()
    assert out.participants == ["mina", "junho"]
    assert out.gifts == [("junho", "mina", "berry")]
    assert out.mean_valence["mina"] == pytest.approx(0.3)


def test_silence_needs_two_turns_and_turn_order():
    s = session()
    s.accept(0)
    s.add_turn(turn(s, 0, end=True), 2)
    assert s.state == "active"  # one-sided end is ignored
    with pytest.raises(ConversationError):
        s.add_turn(ConversationTurn(speaker_id="mina", listener_id="junho", tick=1,
                                    utterance="또 나?"), 2)
    s.add_turn(turn(s, 2, end=True), 2)
    assert s.end_reason == "silence"


def test_decline_and_interrupt():
    s = session()
    s.decline(3)
    assert s.outcome().end_reason == "declined"
    with pytest.raises(ConversationError):
        s.accept(4)
    s2 = session()
    s2.accept(0)
    s2.interrupt(5)
    assert s2.outcome().end_reason == "interrupt"
