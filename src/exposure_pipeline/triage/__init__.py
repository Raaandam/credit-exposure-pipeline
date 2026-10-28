"""LLM-assisted DQ break triage with guardrails, audit logging and human-in-the-loop review.

Modules: ``schemas`` (Pydantic output contract), ``guardrails`` (masking, token budget,
validation, confidence gate), ``context`` (break context assembly), ``chain`` (LangChain +
OpenAI call), ``audit`` (triage_audit + review write-back), ``runner`` (batch CLI), ``eval``.
"""
