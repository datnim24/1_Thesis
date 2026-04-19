from __future__ import annotations

from dataclasses import dataclass


ACTION_COUNT = 21
WAIT_ACTION_ID = 20
RESERVED_ACTION_IDS = (9, 10, 11, 12, 17, 18, 19)


@dataclass(frozen=True)
class ActionDefinition:
    action_id: int
    label: str
    env_action: tuple[str, ...]
    roaster_id: str | None = None
    line_id: str | None = None
    sku: str | None = None
    reserved: bool = False


ACTION_DEFINITIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(0, "PSC on R1 -> L1", ("PSC", "L1"), roaster_id="R1", line_id="L1", sku="PSC"),
    ActionDefinition(1, "PSC on R2 -> L1", ("PSC", "L1"), roaster_id="R2", line_id="L1", sku="PSC"),
    ActionDefinition(2, "PSC on R3 -> L1", ("PSC", "L1"), roaster_id="R3", line_id="L1", sku="PSC"),
    ActionDefinition(3, "PSC on R3 -> L2", ("PSC", "L2"), roaster_id="R3", line_id="L2", sku="PSC"),
    ActionDefinition(4, "PSC on R4 -> L2", ("PSC", "L2"), roaster_id="R4", line_id="L2", sku="PSC"),
    ActionDefinition(5, "PSC on R5 -> L2", ("PSC", "L2"), roaster_id="R5", line_id="L2", sku="PSC"),
    ActionDefinition(6, "NDG on R1", ("NDG",), roaster_id="R1", sku="NDG"),
    ActionDefinition(7, "NDG on R2", ("NDG",), roaster_id="R2", sku="NDG"),
    ActionDefinition(8, "BUSTA on R2", ("BUSTA",), roaster_id="R2", sku="BUSTA"),
    ActionDefinition(9, "Reserved 9", ("WAIT",), reserved=True),
    ActionDefinition(10, "Reserved 10", ("WAIT",), reserved=True),
    ActionDefinition(11, "Reserved 11", ("WAIT",), reserved=True),
    ActionDefinition(12, "Reserved 12", ("WAIT",), reserved=True),
    ActionDefinition(13, "Restock PSC on L1", ("START_RESTOCK", "L1", "PSC"), line_id="L1", sku="PSC"),
    ActionDefinition(14, "Restock NDG on L1", ("START_RESTOCK", "L1", "NDG"), line_id="L1", sku="NDG"),
    ActionDefinition(15, "Restock BUSTA on L1", ("START_RESTOCK", "L1", "BUSTA"), line_id="L1", sku="BUSTA"),
    ActionDefinition(16, "Restock PSC on L2", ("START_RESTOCK", "L2", "PSC"), line_id="L2", sku="PSC"),
    ActionDefinition(17, "Reserved 17", ("WAIT",), reserved=True),
    ActionDefinition(18, "Reserved 18", ("WAIT",), reserved=True),
    ActionDefinition(19, "Reserved 19", ("WAIT",), reserved=True),
    ActionDefinition(20, "WAIT", ("WAIT",)),
)

ACTION_BY_ID = {action.action_id: action for action in ACTION_DEFINITIONS}
ROASTER_ACTION_IDS = {
    "R1": (0, 6),
    "R2": (1, 7, 8),
    "R3": (2, 3),
    "R4": (4,),
    "R5": (5,),
}
RESTOCK_ACTION_IDS = (13, 14, 15, 16)
