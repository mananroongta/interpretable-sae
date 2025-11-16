from __future__ import annotations

from rllm.agents.math_agent import MathAgent


class HarmRLVRAgent(MathAgent):
    """MathAgent with accumulate_thinking turned off"""

    def __init__(self, accumulate_thinking: bool = False):
        super().__init__(accumulate_thinking=accumulate_thinking)
