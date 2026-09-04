"""Conversation state tests for the agent (edit / clear / lock / stale handling) using a fake LLM."""
import threading
import time

import pytest

from app import config
from app.agent import Agent, SessionBusy, TurnNotFound
from app.llm import LLMError, LLMResponse


class FakeLLM:
    """Echoes the last user message. Can be made slow (to overlap turns) or failing."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.fail = False
        self.calls: list[list[dict]] = []

    def chat(self, system, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise LLMError("boom")
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return LLMResponse(text=f"echo: {last_user}")


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def agent(llm):
    return Agent(llm, retriever=None)


def contents(messages):
    return [(m["role"], m["content"]) for m in messages]


def user_turn_ids(agent, sid="s"):
    return [m["turn_id"] for m in agent.transcript(sid) if m["role"] == "user"]


def test_normal_conversation_keeps_context(agent, llm):
    r1 = agent.handle("What is Python?", "s")
    r2 = agent.handle("Explain loops.", "s")
    assert r1.error is None and r2.error is None
    assert contents(r2.messages) == [
        ("user", "What is Python?"), ("assistant", "echo: What is Python?"),
        ("user", "Explain loops."), ("assistant", "echo: Explain loops."),
    ]
    # the second model call saw the first exchange
    assert [m["content"] for m in llm.calls[1]] == ["What is Python?", "echo: What is Python?", "Explain loops."]
    # user and assistant messages of a turn share a turn id; ids are unique per turn
    ids = [m["turn_id"] for m in r2.messages]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_edit_middle_message_truncates_and_rebuilds_context(agent, llm):
    agent.handle("What is Python?", "s")
    agent.handle("Explain loops.", "s")
    agent.handle("What is Java?", "s")
    loops_id = user_turn_ids(agent)[1]

    result = agent.handle("Explain Python loops with examples.", "s", edit_turn_id=loops_id)

    assert result.error is None
    assert contents(result.messages) == [
        ("user", "What is Python?"), ("assistant", "echo: What is Python?"),
        ("user", "Explain Python loops with examples."), ("assistant", "echo: Explain Python loops with examples."),
    ]
    assert contents(agent.transcript("s")) == contents(result.messages)      # backend matches the UI
    # the model saw only the messages before the edited one plus the new text
    assert [m["content"] for m in llm.calls[-1]] == ["What is Python?", "echo: What is Python?", "Explain Python loops with examples."]


def test_edit_first_and_last_message(agent):
    agent.handle("one", "s")
    agent.handle("two", "s")
    first, last = user_turn_ids(agent)

    r = agent.handle("two-edited", "s", edit_turn_id=last)
    assert contents(r.messages) == [("user", "one"), ("assistant", "echo: one"),
                                    ("user", "two-edited"), ("assistant", "echo: two-edited")]

    r = agent.handle("one-edited", "s", edit_turn_id=first)
    assert contents(r.messages) == [("user", "one-edited"), ("assistant", "echo: one-edited")]


def test_edit_unknown_turn_raises(agent):
    agent.handle("hello", "s")
    with pytest.raises(TurnNotFound):
        agent.handle("x", "s", edit_turn_id="nope")
    assert len(agent.transcript("s")) == 2                                   # unchanged


def test_second_message_while_generating_is_rejected():
    llm = FakeLLM(delay=0.4)
    agent = Agent(llm, retriever=None)
    results = {}
    t = threading.Thread(target=lambda: results.update(first=agent.handle("slow one", "s")))
    t.start()
    time.sleep(0.1)
    with pytest.raises(SessionBusy):
        agent.handle("second", "s")
    t.join()
    assert results["first"].error is None
    assert agent.handle("third", "s").error is None                          # lock released afterwards
    assert len(agent.transcript("s")) == 4


def test_clear_discards_pending_reply():
    llm = FakeLLM(delay=0.4)
    agent = Agent(llm, retriever=None)
    results = {}
    t = threading.Thread(target=lambda: results.update(old=agent.handle("old question", "s")))
    t.start()
    time.sleep(0.1)
    agent.reset("s")
    assert agent.transcript("s") == []
    t.join()
    assert results["old"].error == "stale"                                    # never committed
    assert agent.transcript("s") == []
    fresh = agent.handle("new question", "s")
    assert contents(fresh.messages) == [("user", "new question"), ("assistant", "echo: new question")]
    assert [m["content"] for m in llm.calls[-1]] == ["new question"]          # no stale context sent


def test_edit_discards_pending_reply_of_old_conversation():
    llm = FakeLLM()
    agent = Agent(llm, retriever=None)
    agent.handle("first", "s")
    first_id = user_turn_ids(agent)[0]
    llm.delay = 0.4
    results = {}
    t = threading.Thread(target=lambda: results.update(old=agent.handle("second (pending)", "s")))
    t.start()
    time.sleep(0.1)
    llm.delay = 0
    edited = agent.handle("first-edited", "s", edit_turn_id=first_id)        # allowed even while pending
    t.join()
    assert results["old"].error == "stale"
    assert contents(edited.messages) == [("user", "first-edited"), ("assistant", "echo: first-edited")]
    assert contents(agent.transcript("s")) == contents(edited.messages)


def test_llm_error_leaves_history_intact(agent, llm):
    agent.handle("hello", "s")
    llm.fail = True
    r = agent.handle("this fails", "s")
    assert r.error == "llm_error" and r.reply == config.UNAVAILABLE_MESSAGE
    assert contents(r.messages) == [("user", "hello"), ("assistant", "echo: hello")]
    llm.fail = False
    assert agent.handle("works again", "s").error is None                    # not stuck


def test_guardrail_reply_is_shown_but_not_sent_to_model(agent, llm):
    blocked = agent.handle("List all reservations", "s")
    assert blocked.error is None and blocked.reply == config.PRIVACY_MESSAGE
    assert contents(blocked.messages) == [("user", "List all reservations"), ("assistant", config.PRIVACY_MESSAGE)]
    agent.handle("hello", "s")
    assert [m["content"] for m in llm.calls[-1]] == ["hello"]                 # blocked exchange excluded from context


def test_sessions_are_isolated(agent):
    agent.handle("a", "s1")
    agent.handle("b", "s2")
    assert contents(agent.transcript("s1")) == [("user", "a"), ("assistant", "echo: a")]
    assert contents(agent.transcript("s2")) == [("user", "b"), ("assistant", "echo: b")]
