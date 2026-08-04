"""
DAG registration HTTP endpoints for vLLM.

POST  /v1/dags                                    — register a DAG structure
                                                      (nodes + edges).
GET   /v1/dags                                    — list registered DAG IDs.
PATCH /v1/dags/{dag_id}/nodes/{node_id}           — update a node's expected
                                                      token cost with an online
                                                      LLM self-estimate.
POST  /v1/dags/{dag_id}/nodes/{parent}/next-round — append a runtime-discovered
                                                      round node after `parent`
                                                      (for subagents whose total
                                                      round count is unknown
                                                      at DAG-registration time).
POST  /v1/dags/{dag_id}/nodes/{node_id}/complete  — mark a node as finished
                                                      (no further rounds will
                                                      follow it); required on
                                                      the LAST round of a
                                                      dynamically-extended
                                                      subagent so it leaves the
                                                      frontier set.

After registration, chat completion requests can set dag_id + node_id instead
of a raw priority value; the serving layer computes cp_remaining-based priority
automatically from the registry.

Two-layer token prediction flow (subagent with unknown round count):
  1. Register only the subagent's first round as a node at DAG-registration
     time (Layer-1 static estimate as expected_tokens).
  2. Before each round's completion request, if Layer-2's proportional
     extrapolation flags a significant overrun, PATCH the current node's
     expected_tokens.
  3. After a round completes and the subagent decides to continue (another
     tool call needs a follow-up LLM turn), POST .../next-round with a new
     node_id for the upcoming round — BEFORE submitting that round's actual
     completion request, so cp_remaining already reflects the new node when
     the server assigns scheduling priority.
  4. Submit the actual completion request with dag_id + the new node_id.
  5. Repeat steps 2-4 until the subagent's task is done; the final round's
     node never gets a next-round call, so the client MUST explicitly POST
     .../complete on it — otherwise it never leaves the registry's frontier
     set and dag_remaining_cp() keeps over-counting this branch as pending.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from vllm import dag_registry
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


class DAGNode(BaseModel):
    node_id: str
    expected_tokens: int = Field(default=100, ge=1)


class DAGRegistrationRequest(BaseModel):
    dag_id: str
    nodes: list[DAGNode]
    edges: list[tuple[str, str]] = Field(default_factory=list)


class DAGRegistrationResponse(BaseModel):
    dag_id: str
    num_nodes: int
    registered: bool = True


@router.post("/v1/dags", response_model=DAGRegistrationResponse)
async def register_dag(request: DAGRegistrationRequest) -> DAGRegistrationResponse:
    nodes = {n.node_id: n.expected_tokens for n in request.nodes}
    dag_registry.register(request.dag_id, nodes, request.edges)
    logger.info("Registered DAG '%s' with %d nodes", request.dag_id, len(nodes))
    return DAGRegistrationResponse(dag_id=request.dag_id, num_nodes=len(nodes))


@router.get("/v1/dags")
async def list_dags() -> dict[str, list[str]]:
    return {"dag_ids": dag_registry.list_dags()}


class NodeCostUpdate(BaseModel):
    expected_tokens: int = Field(ge=1, description="LLM-estimated output token count for this node")


class NodeCostUpdateResponse(BaseModel):
    dag_id: str
    node_id: str
    expected_tokens: int
    updated: bool = True


@router.patch("/v1/dags/{dag_id}/nodes/{node_id}", response_model=NodeCostUpdateResponse)
async def update_node_cost(
    dag_id: str, node_id: str, body: NodeCostUpdate
) -> NodeCostUpdateResponse:
    """
    Update a node's expected token cost with an online LLM self-estimate.
    Clears the cp_remaining cache so the next request for this (dag_id, node_id)
    uses the updated cost when computing scheduling priority.
    """
    updated = dag_registry.update_node_cost(dag_id, node_id, body.expected_tokens)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"DAG '{dag_id}' or node '{node_id}' not found in registry",
        )
    logger.info(
        "Updated '%s'/'%s' expected_tokens → %d", dag_id, node_id, body.expected_tokens
    )
    return NodeCostUpdateResponse(
        dag_id=dag_id, node_id=node_id, expected_tokens=body.expected_tokens
    )


class AppendRoundRequest(BaseModel):
    new_node_id: str = Field(description="Identifier for the newly discovered round node")
    expected_tokens: int = Field(
        default=1200, ge=1,
        description="Layer-1/Layer-2 token estimate for this newly discovered round",
    )


class AppendRoundResponse(BaseModel):
    dag_id: str
    parent_node_id: str
    new_node_id: str
    expected_tokens: int
    appended: bool = True


@router.post(
    "/v1/dags/{dag_id}/nodes/{parent_node_id}/next-round",
    response_model=AppendRoundResponse,
)
async def append_round(
    dag_id: str, parent_node_id: str, body: AppendRoundRequest
) -> AppendRoundResponse:
    """
    Register the next round of a subagent's conversation as a new DAG node,
    discovered only at runtime (the total round count is unknown upfront).

    Splices new_node_id between parent_node_id and parent's existing
    successors, so downstream join nodes still wait correctly. parent_node_id
    is marked completed; new_node_id becomes the new frontier node for this
    branch's scheduling priority computation.

    Call this once per round, right before submitting the actual completion
    request for that round — so the registry's cp_remaining computation
    already reflects the new node when priority is assigned.
    """
    appended = dag_registry.append_round(
        dag_id, parent_node_id, body.new_node_id, body.expected_tokens
    )
    if not appended:
        raise HTTPException(
            status_code=404,
            detail=(
                f"DAG '{dag_id}' not found, parent node '{parent_node_id}' not "
                f"found, or node '{body.new_node_id}' already exists"
            ),
        )
    logger.info(
        "Appended round: '%s'/'%s' → '%s' (expected_tokens=%d)",
        dag_id, parent_node_id, body.new_node_id, body.expected_tokens,
    )
    return AppendRoundResponse(
        dag_id=dag_id,
        parent_node_id=parent_node_id,
        new_node_id=body.new_node_id,
        expected_tokens=body.expected_tokens,
    )


class MarkCompletedResponse(BaseModel):
    dag_id: str
    node_id: str
    completed: bool = True


@router.post(
    "/v1/dags/{dag_id}/nodes/{node_id}/complete",
    response_model=MarkCompletedResponse,
)
async def mark_node_completed(dag_id: str, node_id: str) -> MarkCompletedResponse:
    """
    Mark a node as completed (no further rounds will follow it).

    Call this once a subagent's LAST round finishes — i.e. when it decides
    NOT to continue (no append_round call follows).  Intermediate rounds
    don't need this call: append_round() already marks the previous round's
    node_id as completed when splicing in the next one.

    Without this call on the final round, the node never leaves the
    registry's frontier set, and dag_remaining_cp() will keep treating this
    branch as still-pending — inflating this DAG's scheduling priority
    relative to other DAGs that have genuinely finished fewer nodes.
    """
    completed = dag_registry.mark_completed(dag_id, node_id)
    if not completed:
        raise HTTPException(
            status_code=404,
            detail=f"DAG '{dag_id}' or node '{node_id}' not found in registry",
        )
    logger.info("Marked completed: '%s'/'%s'", dag_id, node_id)
    return MarkCompletedResponse(dag_id=dag_id, node_id=node_id)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
