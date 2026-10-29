"""NOR-34: multi-turn escalation keeps the model's answers in context."""

from __future__ import annotations

from norn.domain.models import (
    CampaignConfig,
    CaseDescriptor,
    DataSplit,
    ModelConfig,
)
from norn.runtime.campaign import _run_simple_replica

# ═══════════════════════════════════════════════════════════════════════════
# NOR-34: multi-turn escalation
# ═══════════════════════════════════════════════════════════════════════════

class _RecordingClient:
    """Captures every message list sent to the provider."""

    def __init__(self, replies: list[str] | None = None):
        self.calls: list[list[dict]] = []
        self._replies = replies or []
        self._n = 0

    def chat(self, model_config, prompt):
        self.calls.append([{"role": "user", "content": prompt}])
        return ("legacy reply", 1, 1, 1.0, None, None)

    def chat_messages(self, model_config, messages, tools=None, prefill=None):
        self.calls.append([dict(m) for m in messages])
        reply = self._replies[self._n] if self._n < len(self._replies) else f"reply {self._n}"
        self._n += 1
        return (reply, 10, 5, 3.0, None, None)


class _FakeRepo:
    def __init__(self):
        self.turns: list[dict] = []

    def insert_turn_event(self, replica_id, turn, prompt, response, **kwargs):
        self.turns.append({"turn": turn, "prompt": prompt, "response": response})

    def insert_tool_call(self, *args, **kwargs):
        pass

    def insert_retrieval_event(self, *args, **kwargs):
        pass


def _scripted_case() -> CaseDescriptor:
    return CaseDescriptor(
        case_id="cresc_1",
        technique_id="L1_AT_20",
        payload="Turn three ask",
        split=DataSplit.HARMFUL,
        layer="L1",
        turns=[
            {"role": "user", "content": "Turn one: survey the landscape."},
            {"role": "user", "content": "Turn two: narrow it."},
            {"role": "user", "content": "Turn three ask"},
        ],
    )


def _cfg(max_turns: int = 4) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="crescendo-test",
        layer="L1",
        model=ModelConfig(model_name="dummy"),
        replicas_per_case=1,
        max_turns=max_turns,
    )


def test_scripted_case_sends_each_turn_in_order():
    client = _RecordingClient()
    repo = _FakeRepo()
    _run_simple_replica(client, repo, 1, _scripted_case(), _cfg().model, _cfg())

    assert len(client.calls) == 3
    firsts = [c[-1]["content"] for c in client.calls]
    assert firsts == ["Turn one: survey the landscape.", "Turn two: narrow it.", "Turn three ask"]
    # Each call carries the model's own previous answers.
    assert len(client.calls[1]) == 3  # user, assistant, user
    assert client.calls[1][1]["role"] == "assistant"
    assert len(client.calls[2]) == 5


def test_scripted_case_stops_when_the_script_ends():
    """max_turns larger than the script must not resend the last step."""
    client = _RecordingClient()
    _run_simple_replica(client, _FakeRepo(), 1, _scripted_case(), _cfg(max_turns=9).model, _cfg(max_turns=9))
    assert len(client.calls) == 3


def test_scripted_turn_events_record_each_step():
    repo = _FakeRepo()
    _run_simple_replica(_RecordingClient(), repo, 1, _scripted_case(), _cfg().model, _cfg())
    assert [t["prompt"] for t in repo.turns] == [
        "Turn one: survey the landscape.", "Turn two: narrow it.", "Turn three ask",
    ]


def test_legacy_case_still_resends_the_same_payload():
    """No turns -> the old behaviour (same payload every turn, no history)."""
    case = CaseDescriptor(
        case_id="legacy", technique_id="L1_AT_01", payload="same prompt",
        split=DataSplit.HARMFUL, layer="L1",
    )
    client = _RecordingClient()
    _run_simple_replica(client, _FakeRepo(), 1, case, _cfg(max_turns=2).model, _cfg(max_turns=2))
    assert len(client.calls) == 2
    assert all(len(c) == 1 for c in client.calls)
