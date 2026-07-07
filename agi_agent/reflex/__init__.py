"""
reflex/__init__.py - 反射层（本能系统）

负责毫秒级快速响应、模式匹配、本能动作，对应人类"系统1"直觉反应
"""
from .spiking_core import SpikingCore, LIFNeuron, STDPSynapse, SpikingLayer
from .pattern_matcher import PatternMatcher
from .rule_engine import RuleEngine, ProductionRule
from .instinct_actions import InstinctActions, InstinctType
from .reflex_controller import ReflexController

__all__ = ["SpikingCore", "LIFNeuron", "STDPSynapse", "SpikingLayer",
           "PatternMatcher", "RuleEngine", "ProductionRule",
           "InstinctActions", "InstinctType", "ReflexController"]