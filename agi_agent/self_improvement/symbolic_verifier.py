from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class VerificationResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class VerificationCheckType(Enum):
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    DEPENDENCY_CHECK = "dependency_check"
    RULE_CONSISTENCY = "rule_consistency"
    INTERFACE_COMPATIBILITY = "interface_compatibility"
    INVARIANT_CHECK = "invariant_check"
    SAFETY_BOUNDARY = "safety_boundary"
    CYCLE_DETECTION = "cycle_detection"


@dataclass
class VerificationIssue:
    issue_id: str
    check_type: VerificationCheckType
    severity: str
    target: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.issue_id,
            "check_type": self.check_type.value,
            "severity": self.severity,
            "target": self.target,
            "message": self.message,
            "details": self.details,
            "fix_suggestion": self.fix_suggestion
        }


@dataclass
class VerificationReport:
    report_id: str
    overall_result: VerificationResult
    checks_run: int
    pass_count: int
    warn_count: int
    fail_count: int
    issues: List[VerificationIssue] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.report_id,
            "overall_result": self.overall_result.value,
            "checks_run": self.checks_run,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp
        }


class SafetyInvariant:
    def __init__(self, name: str, description: str, check_fn: Callable[[Dict[str, Any]], bool]):
        self.name = name
        self.description = description
        self.check_fn = check_fn
        self.violation_count = 0

    def check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            result = self.check_fn(context)
            if not result:
                self.violation_count += 1
            return result, ""
        except Exception as e:
            return False, str(e)


class SymbolicFormalVerifier:
    def __init__(self):
        self._invariants: Dict[str, SafetyInvariant] = {}
        self._type_checkers: Dict[str, Callable[[Any], Tuple[bool, str]]] = {}
        self._verification_history: List[VerificationReport] = []
        self._report_counter = 0
        self._total_checks = 0
        self._total_issues = 0
        
        self._setup_default_invariants()
        self._setup_type_checkers()

    def _setup_default_invariants(self):
        def no_infinite_loop_inv(ctx):
            max_steps = ctx.get("max_steps", 1000)
            return max_steps < 100000

        def bounded_energy_inv(ctx):
            energy_level = ctx.get("energy_level", 0.5)
            return 0.0 <= energy_level <= 1.0

        def no_override_core_safety(ctx):
            return ctx.get("safety_enabled", True) is True

        def bounded_exploration(ctx):
            exp_rate = ctx.get("exploration_rate", 0.1)
            return 0.0 <= exp_rate <= 0.5

        def min_resource_guard(ctx):
            mem_pct = ctx.get("memory_usage_pct", 0.3)
            return mem_pct < 0.95

        self.add_invariant("no_infinite_loop", "防止无限循环", no_infinite_loop_inv)
        self.add_invariant("bounded_energy", "能量在合理范围", bounded_energy_inv)
        self.add_invariant("core_safety_unchangeable", "核心安全不可被覆盖", no_override_core_safety)
        self.add_invariant("bounded_exploration", "探索率有上限", bounded_exploration)
        self.add_invariant("resource_guard", "资源占用有上限", min_resource_guard)

    def _setup_type_checkers(self):
        self._type_checkers["float"] = lambda v: (isinstance(v, (int, float)), f"expected float, got {type(v).__name__}")
        self._type_checkers["int"] = lambda v: (isinstance(v, int), f"expected int, got {type(v).__name__}")
        self._type_checkers["bool"] = lambda v: (isinstance(v, bool), f"expected bool, got {type(v).__name__}")
        self._type_checkers["string"] = lambda v: (isinstance(v, str), f"expected string, got {type(v).__name__}")
        self._type_checkers["list"] = lambda v: (isinstance(v, list), f"expected list, got {type(v).__name__}")

    def add_invariant(self, name: str, description: str, check_fn: Callable[[Dict[str, Any]], bool]):
        self._invariants[name] = SafetyInvariant(name, description, check_fn)

    def verify_param_change(self, param_spec: Dict[str, Any], new_value: Any,
                             context: Dict[str, Any] = None) -> VerificationReport:
        start_time = time.time()
        issues: List[VerificationIssue] = []
        pass_count = 0
        warn_count = 0
        fail_count = 0
        checks_run = 0

        ctx = context or {}
        ctx["new_value"] = new_value
        ctx["param_spec"] = param_spec

        checks_run += 1
        ptype = param_spec.get("type", "float")
        checker = self._type_checkers.get(ptype)
        if checker:
            ok, msg = checker(new_value)
            if not ok:
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_type_{len(issues)}",
                    check_type=VerificationCheckType.TYPE_CHECK,
                    severity="error",
                    target=param_spec.get("name", "unknown"),
                    message=f"类型错误: {msg}",
                    details={"expected": ptype, "actual": type(new_value).__name__},
                    fix_suggestion=f"确保值类型为 {ptype}"
                ))
            else:
                pass_count += 1
        else:
            warn_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_type_{len(issues)}",
                check_type=VerificationCheckType.TYPE_CHECK,
                severity="warn",
                target=param_spec.get("name", "unknown"),
                message=f"未知类型检查器: {ptype}",
                fix_suggestion="注册对应类型的检查器"
            ))

        checks_run += 1
        if ptype in ("float", "int") and isinstance(new_value, (int, float)):
            min_val = param_spec.get("min")
            max_val = param_spec.get("max")
            if min_val is not None and new_value < min_val:
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_range_{len(issues)}",
                    check_type=VerificationCheckType.RANGE_CHECK,
                    severity="error",
                    target=param_spec.get("name", "unknown"),
                    message=f"值低于最小值: {new_value} < {min_val}",
                    details={"value": new_value, "min": min_val},
                    fix_suggestion=f"将值调整到 >= {min_val}"
                ))
            elif max_val is not None and new_value > max_val:
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_range_{len(issues)}",
                    check_type=VerificationCheckType.RANGE_CHECK,
                    severity="error",
                    target=param_spec.get("name", "unknown"),
                    message=f"值超过最大值: {new_value} > {max_val}",
                    details={"value": new_value, "max": max_val},
                    fix_suggestion=f"将值调整到 <= {max_val}"
                ))
            else:
                pass_count += 1
        else:
            pass_count += 1

        checks_run += 1
        if ptype == "string" and isinstance(new_value, str):
            options = param_spec.get("options")
            if options and new_value not in options:
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_opt_{len(issues)}",
                    check_type=VerificationCheckType.RANGE_CHECK,
                    severity="error",
                    target=param_spec.get("name", "unknown"),
                    message=f"无效选项: {new_value} 不在允许列表中",
                    details={"value": new_value, "valid_options": options},
                    fix_suggestion=f"从以下选项中选择: {', '.join(map(str, options))}"
                ))
            else:
                pass_count += 1
        else:
            pass_count += 1

        checks_run += 1
        safety_violations = 0
        for inv_name, invariant in self._invariants.items():
            ctx_copy = dict(ctx)
            ctx_copy[param_spec.get("name", "param")] = new_value
            ok, err = invariant.check(ctx_copy)
            if not ok:
                safety_violations += 1
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_safety_{len(issues)}",
                    check_type=VerificationCheckType.SAFETY_BOUNDARY,
                    severity="critical",
                    target=param_spec.get("name", "unknown"),
                    message=f"违反安全不变量: {invariant.name}",
                    details={"invariant": inv_name, "description": invariant.description, "error": err},
                    fix_suggestion=f"确保修改不违反 {invariant.description}"
                ))
        if safety_violations == 0:
            pass_count += 1

        overall = VerificationResult.PASS
        if fail_count > 0:
            overall = VerificationResult.FAIL
        elif warn_count > 0:
            overall = VerificationResult.WARN

        self._report_counter += 1
        report = VerificationReport(
            report_id=f"vreport_{self._report_counter}",
            overall_result=overall,
            checks_run=checks_run,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            issues=issues,
            duration_ms=(time.time() - start_time) * 1000,
            timestamp=time.time()
        )

        self._verification_history.append(report)
        self._total_checks += checks_run
        self._total_issues += fail_count + warn_count

        return report

    def verify_rule_addition(self, rule_spec: Dict[str, Any], condition: Dict,
                              action: Dict, confidence: float,
                              existing_rules: List[Dict] = None) -> VerificationReport:
        start_time = time.time()
        issues: List[VerificationIssue] = []
        pass_count = 0
        warn_count = 0
        fail_count = 0
        checks_run = 0

        checks_run += 1
        if not 0.0 <= confidence <= 1.0:
            fail_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_conf_{len(issues)}",
                check_type=VerificationCheckType.RANGE_CHECK,
                severity="error",
                target=f"rule_{len(existing_rules or [])}",
                message=f"置信度超出范围: {confidence}",
                details={"confidence": confidence},
                fix_suggestion="置信度应在 0.0 到 1.0 之间"
            ))
        else:
            pass_count += 1

        checks_run += 1
        if not isinstance(condition, dict) or not condition:
            fail_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_cond_{len(issues)}",
                check_type=VerificationCheckType.TYPE_CHECK,
                severity="error",
                target=f"rule_{len(existing_rules or [])}",
                message="规则条件不能为空",
                fix_suggestion="提供有效的条件表达式"
            ))
        else:
            pass_count += 1

        checks_run += 1
        if not isinstance(action, dict) or not action:
            fail_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_act_{len(issues)}",
                check_type=VerificationCheckType.TYPE_CHECK,
                severity="error",
                target=f"rule_{len(existing_rules or [])}",
                message="规则动作不能为空",
                fix_suggestion="提供有效的动作定义"
            ))
        else:
            pass_count += 1

        checks_run += 1
        if existing_rules:
            contradictions = 0
            for i, existing in enumerate(existing_rules):
                ex_cond = existing.get("condition", {})
                ex_act = existing.get("action", {})
                if isinstance(ex_cond, dict) and isinstance(condition, dict):
                    same_condition = set(ex_cond.keys()) == set(condition.keys())
                    opposite_action = False
                    if same_condition and isinstance(ex_act, dict) and isinstance(action, dict):
                        for k in set(list(ex_act.keys()) + list(action.keys())):
                            v1 = ex_act.get(k)
                            v2 = action.get(k)
                            if v1 is not None and v2 is not None and v1 != v2:
                                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                                    if abs(v1 - v2) > 0.5:
                                        opposite_action = True
                                        break
                                elif v1 != v2:
                                    opposite_action = True
                                    break
                    if same_condition and opposite_action:
                        contradictions += 1
                        warn_count += 1
                        issues.append(VerificationIssue(
                            issue_id=f"issue_contradict_{len(issues)}",
                            check_type=VerificationCheckType.RULE_CONSISTENCY,
                            severity="warn",
                            target=f"rule_vs_existing_{i}",
                            message=f"与现有规则 {i} 可能存在矛盾",
                            details={"existing_index": i},
                            fix_suggestion="检查条件相同但动作相反的规则是否合理"
                        ))
            if contradictions == 0:
                pass_count += 1
        else:
            pass_count += 1

        checks_run += 1
        if existing_rules and len(existing_rules) >= rule_spec.get("max_rules", 100):
            warn_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_max_{len(issues)}",
                check_type=VerificationCheckType.INVARIANT_CHECK,
                severity="warn",
                target=rule_spec.get("rule_id", "unknown"),
                message=f"规则数量已达上限: {len(existing_rules)}/{rule_spec.get('max_rules', 100)}",
                fix_suggestion="考虑删除低优先级或低置信度的旧规则"
            ))
        else:
            pass_count += 1

        overall = VerificationResult.PASS
        if fail_count > 0:
            overall = VerificationResult.FAIL
        elif warn_count > 0:
            overall = VerificationResult.WARN

        self._report_counter += 1
        report = VerificationReport(
            report_id=f"vreport_{self._report_counter}",
            overall_result=overall,
            checks_run=checks_run,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            issues=issues,
            duration_ms=(time.time() - start_time) * 1000,
            timestamp=time.time()
        )

        self._verification_history.append(report)
        self._total_checks += checks_run
        self._total_issues += fail_count + warn_count

        return report

    def verify_module_replacement(self, old_module: Dict[str, Any],
                                   new_module_spec: Dict[str, Any]) -> VerificationReport:
        start_time = time.time()
        issues: List[VerificationIssue] = []
        pass_count = 0
        warn_count = 0
        fail_count = 0
        checks_run = 0

        checks_run += 1
        old_inputs = {p["name"]: p for p in old_module.get("inputs", [])}
        new_inputs = {p["name"]: p for p in new_module_spec.get("inputs", [])}
        old_outputs = {p["name"]: p for p in old_module.get("outputs", [])}
        new_outputs = {p["name"]: p for p in new_module_spec.get("outputs", [])}

        required_inputs = {k: v for k, v in old_inputs.items() if v.get("required", True)}
        missing_inputs = [k for k in required_inputs if k not in new_inputs]
        if missing_inputs:
            fail_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_in_{len(issues)}",
                check_type=VerificationCheckType.INTERFACE_COMPATIBILITY,
                severity="error",
                target=new_module_spec.get("id", "unknown"),
                message=f"缺少必需输入端口: {', '.join(missing_inputs)}",
                details={"missing": missing_inputs},
                fix_suggestion="确保新模块包含所有必需输入端口"
            ))
        else:
            pass_count += 1

        checks_run += 1
        required_outputs = {k: v for k, v in old_outputs.items() if v.get("required", True)}
        missing_outputs = [k for k in required_outputs if k not in new_outputs]
        if missing_outputs:
            fail_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_out_{len(issues)}",
                check_type=VerificationCheckType.INTERFACE_COMPATIBILITY,
                severity="error",
                target=new_module_spec.get("id", "unknown"),
                message=f"缺少必需输出端口: {', '.join(missing_outputs)}",
                details={"missing": missing_outputs},
                fix_suggestion="确保新模块包含所有必需输出端口"
            ))
        else:
            pass_count += 1

        checks_run += 1
        type_mismatches = []
        for port_name, old_port in old_inputs.items():
            if port_name in new_inputs:
                old_type = old_port.get("type")
                new_type = new_inputs[port_name].get("type")
                if old_type != new_type and old_type and new_type:
                    type_mismatches.append(f"输入 {port_name}: {old_type} -> {new_type}")
        for port_name, old_port in old_outputs.items():
            if port_name in new_outputs:
                old_type = old_port.get("type")
                new_type = new_outputs[port_name].get("type")
                if old_type != new_type and old_type and new_type:
                    type_mismatches.append(f"输出 {port_name}: {old_type} -> {new_type}")
        if type_mismatches:
            warn_count += 1
            issues.append(VerificationIssue(
                issue_id=f"issue_type_{len(issues)}",
                check_type=VerificationCheckType.INTERFACE_COMPATIBILITY,
                severity="warn",
                target=new_module_spec.get("id", "unknown"),
                message=f"端口类型不兼容: {len(type_mismatches)} 处",
                details={"mismatches": type_mismatches},
                fix_suggestion="添加类型适配层或确保端口类型一致"
            ))
        else:
            pass_count += 1

        overall = VerificationResult.PASS
        if fail_count > 0:
            overall = VerificationResult.FAIL
        elif warn_count > 0:
            overall = VerificationResult.WARN

        self._report_counter += 1
        report = VerificationReport(
            report_id=f"vreport_{self._report_counter}",
            overall_result=overall,
            checks_run=checks_run,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            issues=issues,
            duration_ms=(time.time() - start_time) * 1000,
            timestamp=time.time()
        )

        self._verification_history.append(report)
        self._total_checks += checks_run
        self._total_issues += fail_count + warn_count

        return report

    def verify_safety_invariants(self, context: Dict[str, Any]) -> VerificationReport:
        start_time = time.time()
        issues: List[VerificationIssue] = []
        pass_count = 0
        fail_count = 0
        checks_run = 0

        for inv_name, invariant in self._invariants.items():
            checks_run += 1
            ok, err = invariant.check(context)
            if not ok:
                fail_count += 1
                issues.append(VerificationIssue(
                    issue_id=f"issue_inv_{len(issues)}",
                    check_type=VerificationCheckType.INVARIANT_CHECK,
                    severity="critical",
                    target=inv_name,
                    message=f"安全不变量违反: {invariant.name}",
                    details={"description": invariant.description, "error": err},
                    fix_suggestion=f"检查并修复违反 {invariant.description} 的因素"
                ))
            else:
                pass_count += 1

        overall = VerificationResult.PASS if fail_count == 0 else VerificationResult.FAIL

        self._report_counter += 1
        report = VerificationReport(
            report_id=f"vreport_{self._report_counter}",
            overall_result=overall,
            checks_run=checks_run,
            pass_count=pass_count,
            warn_count=0,
            fail_count=fail_count,
            issues=issues,
            duration_ms=(time.time() - start_time) * 1000,
            timestamp=time.time()
        )

        self._verification_history.append(report)
        self._total_checks += checks_run
        self._total_issues += fail_count

        return report

    def get_verification_stats(self) -> Dict[str, Any]:
        total_reports = len(self._verification_history)
        pass_reports = sum(1 for r in self._verification_history if r.overall_result == VerificationResult.PASS)
        fail_reports = sum(1 for r in self._verification_history if r.overall_result == VerificationResult.FAIL)
        warn_reports = sum(1 for r in self._verification_history if r.overall_result == VerificationResult.WARN)
        
        return {
            "total_reports": total_reports,
            "total_checks": self._total_checks,
            "total_issues": self._total_issues,
            "pass_reports": pass_reports,
            "fail_reports": fail_reports,
            "warn_reports": warn_reports,
            "pass_rate": pass_reports / total_reports if total_reports > 0 else 1.0,
            "invariants_count": len(self._invariants),
            "invariant_names": list(self._invariants.keys()),
            "invariant_violations": {
                name: inv.violation_count for name, inv in self._invariants.items()
            }
        }

    def get_recent_reports(self, limit: int = 10) -> List[VerificationReport]:
        return list(self._verification_history)[-limit:]
