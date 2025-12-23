# Integration Test: Human-in-the-Loop Approval

## Overview

Content moderation workflow that pauses for human review before publishing. Demonstrates `InterruptNode` for pause/resume.

---

## Scenario

```
Generate content → [PAUSE: Show to human] → Human approves → Publish
                                          → Human rejects → Revise → [PAUSE] → ...
```

---

## Graph Definition

```python
from hypernodes import Graph, node, route, END, InterruptNode, AsyncRunner

@node(output_name="content")
def generate_content(prompt: str, revision_notes: str | None = None) -> str:
    """Generate or revise content."""
    if revision_notes:
        return f"Revised content addressing: {revision_notes}"
    return f"Generated content for: {prompt}"

# Human review interrupt
human_review = InterruptNode(
    name="human_review",
    input_param="content",        # What to show the human
    response_param="review",      # Where human's response goes
)

@route(targets=["generate_content", "publish", END])
def handle_review(review: dict) -> str:
    """Route based on human's decision."""
    if review["approved"]:
        return "publish"
    elif review.get("abandon"):
        return END
    else:
        return "generate_content"  # Revise

@node(output_name="revision_notes")
def extract_notes(review: dict) -> str:
    """Extract revision notes from rejection."""
    return review.get("feedback", "Please improve")

@node(output_name="published")
def publish(content: str) -> dict:
    """Publish approved content."""
    return {"status": "published", "content": content}

approval_graph = Graph(nodes=[
    generate_content,
    human_review,
    handle_review,
    extract_notes,
    publish,
])
```

---

## Test Cases

### test_interrupt_pauses_execution

```python
@pytest.mark.asyncio
async def test_interrupt_pauses_execution():
    """Execution pauses at InterruptNode."""
    runner = AsyncRunner()
    
    result = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    assert result.interrupted is True
    assert result.interrupt_name == "human_review"
    assert result.checkpoint is not None
```

### test_interrupt_provides_content

```python
@pytest.mark.asyncio
async def test_interrupt_provides_content():
    """Interrupt includes content for human review."""
    runner = AsyncRunner()
    
    result = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    assert result.interrupt_value is not None
    assert "Generated content" in result.interrupt_value
```

### test_resume_with_approval

```python
@pytest.mark.asyncio
async def test_resume_with_approval():
    """Approved content gets published."""
    runner = AsyncRunner()
    
    # First run - hits interrupt
    result1 = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    # Resume with approval
    result2 = await runner.run(
        approval_graph,
        inputs={"review": {"approved": True}},
        checkpoint=result1.checkpoint,
    )
    
    assert result2.interrupted is False
    assert result2.outputs["published"]["status"] == "published"
```

### test_resume_with_rejection

```python
@pytest.mark.asyncio
async def test_resume_with_rejection():
    """Rejected content triggers revision loop."""
    runner = AsyncRunner()
    
    # First run
    result1 = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    # Reject with feedback
    result2 = await runner.run(
        approval_graph,
        inputs={"review": {"approved": False, "feedback": "Add more examples"}},
        checkpoint=result1.checkpoint,
    )
    
    # Should pause again for review of revision
    assert result2.interrupted is True
    assert "Revised content" in result2.interrupt_value
    assert "Add more examples" in result2.interrupt_value
```

### test_multiple_revision_cycles

```python
@pytest.mark.asyncio
async def test_multiple_revision_cycles():
    """Can go through multiple review cycles."""
    runner = AsyncRunner()
    
    # Initial generation
    result = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    # First rejection
    result = await runner.run(
        approval_graph,
        inputs={"review": {"approved": False, "feedback": "revision 1"}},
        checkpoint=result.checkpoint,
    )
    assert result.interrupted is True
    
    # Second rejection
    result = await runner.run(
        approval_graph,
        inputs={"review": {"approved": False, "feedback": "revision 2"}},
        checkpoint=result.checkpoint,
    )
    assert result.interrupted is True
    
    # Finally approve
    result = await runner.run(
        approval_graph,
        inputs={"review": {"approved": True}},
        checkpoint=result.checkpoint,
    )
    assert result.interrupted is False
    assert result.outputs["published"] is not None
```

### test_abandon_workflow

```python
@pytest.mark.asyncio
async def test_abandon_workflow():
    """Human can abandon workflow entirely."""
    runner = AsyncRunner()
    
    result1 = await runner.run(
        approval_graph,
        inputs={"prompt": "Write a blog post"},
    )
    
    # Abandon
    result2 = await runner.run(
        approval_graph,
        inputs={"review": {"approved": False, "abandon": True}},
        checkpoint=result1.checkpoint,
    )
    
    assert result2.interrupted is False
    assert "published" not in result2.outputs
```

---

## Test Cases: Streaming with Interrupts

### test_iter_yields_interrupt_event

```python
@pytest.mark.asyncio
async def test_iter_yields_interrupt_event():
    """iter() yields InterruptEvent when pausing."""
    runner = AsyncRunner()
    events = []
    
    async for event in runner.iter(
        approval_graph,
        inputs={"prompt": "test"},
    ):
        events.append(event)
        if isinstance(event, InterruptEvent):
            break
    
    interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupt_events) == 1
    assert interrupt_events[0].interrupt_name == "human_review"
```

### test_iter_checkpoint_in_event

```python
@pytest.mark.asyncio
async def test_iter_checkpoint_in_event():
    """InterruptEvent contains checkpoint for resume."""
    runner = AsyncRunner()
    
    async for event in runner.iter(
        approval_graph,
        inputs={"prompt": "test"},
    ):
        if isinstance(event, InterruptEvent):
            assert event.checkpoint is not None
            assert len(event.checkpoint) > 0
            break
```

---

## Test Cases: Error Handling

### test_sync_runner_rejects_interrupt

```python
def test_sync_runner_rejects_interrupt():
    """Sync Runner can't handle InterruptNode."""
    runner = Runner()  # Sync!
    
    with pytest.raises(IncompatibleRunnerError) as exc:
        runner.run(approval_graph, inputs={"prompt": "test"})
    
    assert "InterruptNode" in str(exc.value)
    assert "AsyncRunner" in str(exc.value)
```

### test_missing_checkpoint_on_resume

```python
@pytest.mark.asyncio
async def test_missing_checkpoint_on_resume():
    """Resume without checkpoint raises clear error."""
    runner = AsyncRunner()
    
    # Try to resume without checkpoint
    with pytest.raises(ValueError) as exc:
        await runner.run(
            approval_graph,
            inputs={"review": {"approved": True}},
            # checkpoint missing!
        )
    
    # Should explain the issue
    assert "checkpoint" in str(exc.value).lower()
```

### test_corrupted_checkpoint

```python
@pytest.mark.asyncio
async def test_corrupted_checkpoint():
    """Corrupted checkpoint raises CheckpointError."""
    runner = AsyncRunner()
    
    with pytest.raises(CheckpointError):
        await runner.run(
            approval_graph,
            inputs={"review": {"approved": True}},
            checkpoint=b"corrupted data",
        )
```

---

## Acceptance Criteria

- [ ] Execution pauses at InterruptNode
- [ ] Checkpoint is serializable and restorable
- [ ] Human response reaches downstream nodes
- [ ] Can go through multiple review cycles
- [ ] Can abandon workflow at any point
- [ ] iter() yields InterruptEvent with checkpoint
- [ ] Sync Runner gives clear error about incompatibility

---

## Checkpoint Serialization Requirements

The checkpoint must serialize:
- All current values and versions
- Execution history (which nodes ran)
- Gate decisions made
- Position in graph (which node we're paused at)

The checkpoint must NOT serialize:
- Cache state (cache is owned by runner)
- Callbacks
- Function references (only names)
