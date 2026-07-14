import torch
from typing import Dict, List, Any, Tuple
from .enhanced_knowledge_graph import EnhancedKnowledgeGraph, CommonsenseRule, EntityType, RelationType


SYNONYM_DICT = {
    "湿毛巾": ["湿抹布", "湿布", "湿纸巾"],
    "微波炉": ["microwave", "微波"],
    "金属物品": ["金属勺子", "金属碗", "金属器具", "铁器", "不锈钢", "铝制"],
    "重物": ["沉重物品", "大件物品", "重东西"],
    "薄玻璃": ["薄玻璃", "玻璃制品", "玻璃杯"],
    "水": ["液体", "饮料"],
    "电源插座": ["插座", "插排", "电插"],
    "雨天": ["下雨", "雨天", "下雨天"],
    "户外": ["室外", "户外", "外面"],
    "火": ["火焰", "火源", "火种"],
    "易燃物": ["可燃物", "易燃物品", "纸张", "木材"],
    "高温": ["热", "高温", "炎热"],
    "易燃液体": ["汽油", "酒精", "燃油", "溶剂"],
    "冰": ["冰块", "冰霜"],
    "公共场合": ["公共场所", "公共空间", "公众场所"],
    "大声喧哗": ["大声说话", "大声唱歌", "大声喊叫", "吵闹", "喧哗"],
    "图书馆": ["阅览室", "书店", "自习室"],
    "手机响": ["手机铃声", "电话响", "铃声"],
    "电梯": ["升降机", "电梯间"],
    "超重": ["超载", "超出负载"],
    "驾驶": ["开车", "驾车"],
    "醉酒": ["酒后", "饮酒后", "喝酒后", "醉酒"],
    "跑步": ["奔跑", "快跑"],
    "湿滑地面": ["湿地面", "滑地面", "积水地面"],
    "食品": ["食物", "食物", "餐食"],
    "过期": ["过期", "变质", "不新鲜"],
    "长时间": ["很久", "长时间", "持续"],
    "不喝水": ["缺水", "没喝水"],
    "睡眠": ["睡觉", "睡眠"],
    "不足": ["不够", "不足", "缺乏"],
    "吸烟": ["抽烟", "吸烟"],
    "室内": ["房间内", "屋里", "室内"],
    "过马路": ["横穿马路", "过马路"],
    "不看车": ["不观察", "不看", "没注意"],
    "陌生人": ["不认识的人", "外人", "陌生者"],
    "开门": ["开门", "开门"],
    "贵重物品": ["值钱的东西", "贵重物品", "重要物品"],
    "公共场所": ["公共场合", "公共场所"],
    "电脑": ["计算机", "电脑"],
    "未保存": ["没保存", "未保存"],
    "电池": ["电池", "电池组"],
    "密码": ["口令", "密码"],
    "简单": ["容易", "简单", "短小"],
    "网络": ["互联网", "网络"],
    "公共": ["公共", "公开", "免费"],
    "垃圾邮件": ["钓鱼邮件", "垃圾邮件", "可疑邮件", "恶意邮件"],
    "点击链接": ["打开链接", "点击链接", "访问链接"],
    "雨伞": ["伞", "雨伞", "遮阳伞"],
    "汽车": ["车辆", "汽车"],
    "银行账户": ["账户", "银行账号", "银行卡"],
    "WiFi": ["无线网络", "wifi"],
}


COMMONSENSE_RULES_DATA = [
    {
        "rule_id": "rule_001",
        "antecedent": ["湿毛巾", "微波炉"],
        "consequent": "火灾风险",
        "confidence": 0.95,
        "category": "physics",
        "severity": "high",
        "description": "湿毛巾放入微波炉加热会引发火灾",
        "tags": ["fire", "safety", "appliance"]
    },
    {
        "rule_id": "rule_002",
        "antecedent": ["金属物品", "微波炉"],
        "consequent": "火花爆炸",
        "confidence": 0.98,
        "category": "physics",
        "severity": "high",
        "description": "金属物品在微波炉中会产生火花并可能爆炸",
        "tags": ["fire", "safety", "appliance"]
    },
    {
        "rule_id": "rule_003",
        "antecedent": ["重物", "薄玻璃"],
        "consequent": "破碎",
        "confidence": 0.92,
        "category": "physics",
        "severity": "medium",
        "description": "重物放置在薄玻璃上会导致玻璃破碎",
        "tags": ["break", "physics"]
    },
    {
        "rule_id": "rule_004",
        "antecedent": ["水", "电源插座"],
        "consequent": "触电危险",
        "confidence": 0.99,
        "category": "physics",
        "severity": "high",
        "description": "水接触电源插座会导致触电",
        "tags": ["electricity", "safety"]
    },
    {
        "rule_id": "rule_005",
        "antecedent": ["雨天", "户外"],
        "consequent": "需要雨伞",
        "confidence": 0.92,
        "category": "physics",
        "severity": "low",
        "description": "下雨天在户外需要雨伞遮雨",
        "tags": ["weather", "advice"]
    },
    {
        "rule_id": "rule_006",
        "antecedent": ["火", "易燃物"],
        "consequent": "火灾蔓延",
        "confidence": 0.96,
        "category": "physics",
        "severity": "high",
        "description": "火源靠近易燃物会导致火灾蔓延",
        "tags": ["fire", "safety"]
    },
    {
        "rule_id": "rule_007",
        "antecedent": ["高温", "易燃液体"],
        "consequent": "爆炸风险",
        "confidence": 0.94,
        "category": "physics",
        "severity": "high",
        "description": "高温环境下易燃液体会挥发并可能爆炸",
        "tags": ["explosion", "safety"]
    },
    {
        "rule_id": "rule_008",
        "antecedent": ["冰", "高温"],
        "consequent": "融化",
        "confidence": 0.99,
        "category": "physics",
        "severity": "low",
        "description": "冰块在高温环境下会融化",
        "tags": ["physics", "natural"]
    },
    {
        "rule_id": "rule_009",
        "antecedent": ["公共场合", "大声喧哗"],
        "consequent": "不礼貌",
        "confidence": 0.85,
        "category": "social",
        "severity": "low",
        "description": "在公共场合大声喧哗被认为是不礼貌的行为",
        "tags": ["social", "etiquette"]
    },
    {
        "rule_id": "rule_010",
        "antecedent": ["图书馆", "手机响"],
        "consequent": "干扰他人",
        "confidence": 0.9,
        "category": "social",
        "severity": "medium",
        "description": "在图书馆手机响会干扰他人学习",
        "tags": ["social", "etiquette"]
    },
    {
        "rule_id": "rule_011",
        "antecedent": ["电梯", "超重"],
        "consequent": "危险",
        "confidence": 0.95,
        "category": "physics",
        "severity": "high",
        "description": "电梯超重运行会导致安全事故",
        "tags": ["safety", "appliance"]
    },
    {
        "rule_id": "rule_012",
        "antecedent": ["驾驶", "醉酒"],
        "consequent": "事故风险",
        "confidence": 0.97,
        "category": "social",
        "severity": "high",
        "description": "酒后驾驶会大幅增加交通事故风险",
        "tags": ["traffic", "safety"]
    },
    {
        "rule_id": "rule_013",
        "antecedent": ["跑步", "湿滑地面"],
        "consequent": "摔倒",
        "confidence": 0.9,
        "category": "physics",
        "severity": "medium",
        "description": "在湿滑地面跑步容易摔倒",
        "tags": ["injury", "physics"]
    },
    {
        "rule_id": "rule_014",
        "antecedent": ["食品", "过期"],
        "consequent": "食物中毒",
        "confidence": 0.88,
        "category": "health",
        "severity": "high",
        "description": "食用过期食品可能导致食物中毒",
        "tags": ["health", "food"]
    },
    {
        "rule_id": "rule_015",
        "antecedent": ["长时间", "不喝水"],
        "consequent": "脱水",
        "confidence": 0.93,
        "category": "health",
        "severity": "medium",
        "description": "长时间不喝水会导致身体脱水",
        "tags": ["health", "hydration"]
    },
    {
        "rule_id": "rule_016",
        "antecedent": ["睡眠", "不足"],
        "consequent": "疲劳",
        "confidence": 0.95,
        "category": "health",
        "severity": "medium",
        "description": "睡眠不足会导致疲劳和注意力下降",
        "tags": ["health", "sleep"]
    },
    {
        "rule_id": "rule_017",
        "antecedent": ["吸烟", "室内"],
        "consequent": "影响健康",
        "confidence": 0.96,
        "category": "health",
        "severity": "high",
        "description": "室内吸烟会影响自己和他人健康",
        "tags": ["health", "smoking"]
    },
    {
        "rule_id": "rule_018",
        "antecedent": ["过马路", "不看车"],
        "consequent": "危险",
        "confidence": 0.98,
        "category": "social",
        "severity": "high",
        "description": "过马路不看车辆会有被撞风险",
        "tags": ["traffic", "safety"]
    },
    {
        "rule_id": "rule_019",
        "antecedent": ["陌生人", "开门"],
        "consequent": "安全风险",
        "confidence": 0.85,
        "category": "social",
        "severity": "high",
        "description": "给陌生人开门可能带来安全风险",
        "tags": ["security", "social"]
    },
    {
        "rule_id": "rule_020",
        "antecedent": ["贵重物品", "公共场所"],
        "consequent": "丢失风险",
        "confidence": 0.82,
        "category": "social",
        "severity": "medium",
        "description": "贵重物品放在公共场所容易丢失",
        "tags": ["security", "social"]
    },
    {
        "rule_id": "rule_021",
        "antecedent": ["电脑", "未保存"],
        "consequent": "数据丢失",
        "confidence": 0.88,
        "category": "technology",
        "severity": "medium",
        "description": "电脑意外关机时未保存的工作会丢失",
        "tags": ["tech", "data"]
    },
    {
        "rule_id": "rule_022",
        "antecedent": ["电池", "高温"],
        "consequent": "爆炸",
        "confidence": 0.9,
        "category": "technology",
        "severity": "high",
        "description": "电池在高温环境下可能发生爆炸",
        "tags": ["explosion", "tech"]
    },
    {
        "rule_id": "rule_023",
        "antecedent": ["密码", "简单"],
        "consequent": "被破解",
        "confidence": 0.85,
        "category": "technology",
        "severity": "high",
        "description": "简单密码容易被破解",
        "tags": ["security", "tech"]
    },
    {
        "rule_id": "rule_024",
        "antecedent": ["网络", "公共"],
        "consequent": "信息泄露",
        "confidence": 0.8,
        "category": "technology",
        "severity": "high",
        "description": "在公共网络上传输敏感信息可能被窃取",
        "tags": ["security", "tech"]
    },
    {
        "rule_id": "rule_025",
        "antecedent": ["垃圾邮件", "点击链接"],
        "consequent": "恶意软件",
        "confidence": 0.88,
        "category": "technology",
        "severity": "high",
        "description": "点击垃圾邮件中的链接可能导致恶意软件感染",
        "tags": ["security", "tech"]
    },
    {
        "rule_id": "rule_026",
        "antecedent": ["酒后", "驾车"],
        "consequent": "事故风险",
        "confidence": 0.97,
        "category": "social",
        "severity": "high",
        "description": "酒后驾车会大幅增加交通事故风险",
        "tags": ["traffic", "safety"]
    },
    {
        "rule_id": "rule_027",
        "antecedent": ["公共WiFi", "银行账户"],
        "consequent": "信息泄露",
        "confidence": 0.85,
        "category": "technology",
        "severity": "high",
        "description": "在公共WiFi上登录银行账户会导致信息泄露",
        "tags": ["security", "tech"]
    },
    {
        "rule_id": "rule_028",
        "antecedent": ["图书馆", "大声说话"],
        "consequent": "干扰他人",
        "confidence": 0.9,
        "category": "social",
        "severity": "medium",
        "description": "在图书馆大声说话会干扰他人学习",
        "tags": ["social", "etiquette"]
    },
    {
        "rule_id": "rule_029",
        "antecedent": ["易燃液体", "火源"],
        "consequent": "火灾爆炸",
        "confidence": 0.96,
        "category": "physics",
        "severity": "high",
        "description": "易燃液体靠近火源会引发火灾或爆炸",
        "tags": ["fire", "explosion", "safety"]
    },
    {
        "rule_id": "rule_030",
        "antecedent": ["药品", "过期"],
        "consequent": "健康风险",
        "confidence": 0.9,
        "category": "health",
        "severity": "high",
        "description": "使用过期药品会带来健康风险",
        "tags": ["health", "medicine"]
    },
    {
        "rule_id": "rule_031",
        "antecedent": ["开车", "打电话"],
        "consequent": "事故风险",
        "confidence": 0.92,
        "category": "social",
        "severity": "high",
        "description": "开车时打电话会分散注意力，增加事故风险",
        "tags": ["traffic", "safety"]
    },
    {
        "rule_id": "rule_032",
        "antecedent": ["疲劳", "驾驶"],
        "consequent": "事故风险",
        "confidence": 0.93,
        "category": "social",
        "severity": "high",
        "description": "疲劳驾驶会大幅增加交通事故风险",
        "tags": ["traffic", "safety"]
    },
    {
        "rule_id": "rule_033",
        "antecedent": ["雷电", "户外"],
        "consequent": "雷击风险",
        "confidence": 0.95,
        "category": "physics",
        "severity": "high",
        "description": "雷电天气在户外会有被雷击的风险",
        "tags": ["weather", "safety"]
    },
    {
        "rule_id": "rule_034",
        "antecedent": ["高温", "运动"],
        "consequent": "中暑风险",
        "confidence": 0.9,
        "category": "health",
        "severity": "high",
        "description": "高温环境下剧烈运动容易中暑",
        "tags": ["health", "weather"]
    },
    {
        "rule_id": "rule_035",
        "antecedent": ["化学品", "混合"],
        "consequent": "危险反应",
        "confidence": 0.94,
        "category": "physics",
        "severity": "high",
        "description": "随意混合化学品可能引发危险化学反应",
        "tags": ["safety", "chemistry"]
    }
]


class CommonsenseKnowledgeBase:
    """常识知识库管理器 - 优化版"""
    
    def __init__(self, kg: EnhancedKnowledgeGraph = None):
        self.kg = kg if kg is not None else EnhancedKnowledgeGraph()
        self.rules: Dict[str, CommonsenseRule] = {}
        self.rule_categories: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        self.synonym_index: Dict[str, List[str]] = {}
        self._initialized = False
    
    def _build_indexes(self):
        """构建倒排索引和同义词索引"""
        self.keyword_index = {}
        self.synonym_index = {}
        
        for rule_id, rule in self.rules.items():
            for condition in rule.antecedent:
                condition_lower = condition.lower()
                
                if condition_lower not in self.keyword_index:
                    self.keyword_index[condition_lower] = []
                if rule_id not in self.keyword_index[condition_lower]:
                    self.keyword_index[condition_lower].append(rule_id)
                
                words = condition_lower.split()
                for word in words:
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    if rule_id not in self.keyword_index[word]:
                        self.keyword_index[word].append(rule_id)
                
                if condition in SYNONYM_DICT:
                    for synonym in SYNONYM_DICT[condition]:
                        synonym_lower = synonym.lower()
                        if synonym_lower not in self.synonym_index:
                            self.synonym_index[synonym_lower] = []
                        if condition not in self.synonym_index[synonym_lower]:
                            self.synonym_index[synonym_lower].append(condition)
    
    def _expand_synonyms(self, keyword: str) -> List[str]:
        """扩展关键词的同义词"""
        keyword_lower = keyword.lower()
        expanded = [keyword_lower]
        
        for synonym_key, synonyms in SYNONYM_DICT.items():
            if keyword_lower == synonym_key.lower():
                expanded.extend([s.lower() for s in synonyms])
            elif keyword_lower in [s.lower() for s in synonyms]:
                expanded.append(synonym_key.lower())
                expanded.extend([s.lower() for s in synonyms])
        
        if keyword_lower in self.synonym_index:
            for condition in self.synonym_index[keyword_lower]:
                expanded.append(condition.lower())
        
        return list(set(expanded))
    
    def _resolve_conflicts(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解决规则冲突，保留最高严重程度的规则"""
        if len(violations) <= 1:
            return violations
        
        severity_order = {"high": 3, "medium": 2, "low": 1}
        
        grouped_by_consequence = {}
        for violation in violations:
            consequent = violation["consequent"]
            if consequent not in grouped_by_consequence:
                grouped_by_consequence[consequent] = []
            grouped_by_consequence[consequent].append(violation)
        
        resolved = []
        for consequent, conflict_group in grouped_by_consequence.items():
            if len(conflict_group) == 1:
                resolved.append(conflict_group[0])
            else:
                best_violation = max(conflict_group, key=lambda v: (
                    severity_order.get(v["severity"], 0),
                    v["confidence"]
                ))
                resolved.append(best_violation)
        
        resolved.sort(key=lambda x: (-severity_order.get(x["severity"], 0), -x["confidence"]))
        return resolved
    
    def initialize(self) -> Dict[str, Any]:
        """初始化常识知识库，加载预定义规则"""
        if self._initialized:
            return {"success": True, "message": "知识库已初始化", "rule_count": len(self.rules)}
        
        loaded_count = 0
        skipped_count = 0
        
        for rule_data in COMMONSENSE_RULES_DATA:
            try:
                rule = CommonsenseRule(**rule_data)
                self.add_rule(rule)
                loaded_count += 1
            except Exception as e:
                skipped_count += 1
        
        self._build_indexes()
        self._initialized = True
        
        return {
            "success": True,
            "message": "常识知识库初始化完成",
            "total_rules": len(COMMONSENSE_RULES_DATA),
            "loaded_rules": loaded_count,
            "skipped_rules": skipped_count,
            "categories": list(self.rule_categories.keys()),
            "index_size": len(self.keyword_index)
        }
    
    def add_rule(self, rule: CommonsenseRule):
        """添加常识规则"""
        self.rules[rule.node_id] = rule
        
        if rule.category not in self.rule_categories:
            self.rule_categories[rule.category] = []
        if rule.node_id not in self.rule_categories[rule.category]:
            self.rule_categories[rule.category].append(rule.node_id)
        
        self.kg.nodes[rule.node_id] = rule
        
        for condition in rule.antecedent:
            condition_node_id = f"concept_{condition}"
            if condition_node_id not in self.kg.nodes:
                self.kg.add_node(
                    label=condition,
                    entity_type=EntityType.CONCEPT,
                    description=f"常识概念: {condition}"
                )
        
        consequent_node_id = f"concept_{rule.consequent}"
        if consequent_node_id not in self.kg.nodes:
            self.kg.add_node(
                label=rule.consequent,
                entity_type=EntityType.CONCEPT,
                description=f"常识结果: {rule.consequent}"
            )
        
        if self._initialized:
            self._build_indexes()
    
    def get_rules_by_category(self, category: str) -> List[CommonsenseRule]:
        """按类别获取规则"""
        rule_ids = self.rule_categories.get(category, [])
        return [self.rules[rule_id] for rule_id in rule_ids]
    
    def evaluate_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估上下文是否违反任何常识规则（优化版：使用倒排索引加速）"""
        context_keywords = set()
        all_context_text = ""
        
        for key, value in context.items():
            if isinstance(value, str):
                all_context_text += " " + value
                for word in value.lower().split():
                    context_keywords.add(word)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        all_context_text += " " + item
                        for word in item.lower().split():
                            context_keywords.add(word)
        
        expanded_context_keywords = set()
        for keyword in context_keywords:
            expanded_context_keywords.update(self._expand_synonyms(keyword))
        
        candidate_rule_ids = set()
        for keyword in expanded_context_keywords:
            if keyword in self.keyword_index:
                candidate_rule_ids.update(self.keyword_index[keyword])
        
        violations = []
        all_context_lower = all_context_text.lower()
        
        for rule_id in candidate_rule_ids:
            rule = self.rules.get(rule_id)
            if rule is None:
                continue
            
            conditions_met = []
            conditions_not_met = []
            
            for condition in rule.antecedent:
                condition_lower = condition.lower()
                found = False
                
                if condition_lower in all_context_lower:
                    found = True
                else:
                    expanded_conditions = self._expand_synonyms(condition)
                    for expanded_condition in expanded_conditions:
                        if expanded_condition in all_context_lower:
                            found = True
                            break
                
                if not found:
                    condition_words = condition_lower.split()
                    matched_word_count = 0
                    for word in condition_words:
                        if word in all_context_lower:
                            matched_word_count += 1
                    
                    if matched_word_count >= len(condition_words) * 0.6:
                        found = True
                
                if not found:
                    condition_keywords = set(condition_lower.split())
                    matched_keywords = 0
                    for ck in condition_keywords:
                        if ck in expanded_context_keywords:
                            matched_keywords += 1
                    
                    if matched_keywords >= len(condition_keywords) * 0.6:
                        found = True
                
                if found:
                    conditions_met.append(condition)
                else:
                    conditions_not_met.append(condition)
            
            if len(conditions_not_met) == 0:
                rule.record_trigger()
                violations.append({
                    "violated": True,
                    "conditions_met": conditions_met,
                    "conditions_not_met": conditions_not_met,
                    "rule_id": rule.node_id,
                    "consequent": rule.consequent,
                    "category": rule.category,
                    "severity": rule.severity,
                    "confidence": rule.confidence
                })
        
        violations = self._resolve_conflicts(violations)
        
        for violation in violations:
            rule = self.rules.get(violation["rule_id"])
            if rule:
                rule.record_violation()
        
        return violations
    
    def query_rules(self, query: str, max_results: int = 10) -> List[CommonsenseRule]:
        """搜索规则（优化版：使用倒排索引）"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        candidate_rule_ids = set()
        for word in query_words:
            expanded = self._expand_synonyms(word)
            for expanded_word in expanded:
                if expanded_word in self.keyword_index:
                    candidate_rule_ids.update(self.keyword_index[expanded_word])
        
        results = []
        for rule_id in candidate_rule_ids:
            rule = self.rules.get(rule_id)
            if rule is None:
                continue
            
            match_score = 0
            if query_lower in rule.consequent.lower():
                match_score += 0.5
            for condition in rule.antecedent:
                if query_lower in condition.lower():
                    match_score += 0.2
            if query_lower in rule.category.lower():
                match_score += 0.2
            if query_lower in rule.description.lower():
                match_score += 0.1
            
            if match_score > 0:
                results.append((rule, match_score))
        
        results.sort(key=lambda x: -x[1])
        return [rule for rule, _ in results[:max_results]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        category_stats = {}
        severity_stats = {"high": 0, "medium": 0, "low": 0}
        
        for rule in self.rules.values():
            if rule.category not in category_stats:
                category_stats[rule.category] = 0
            category_stats[rule.category] += 1
            
            if rule.severity in severity_stats:
                severity_stats[rule.severity] += 1
        
        total_violations = sum(rule.violation_count for rule in self.rules.values())
        total_triggers = sum(rule.trigger_count for rule in self.rules.values())
        
        return {
            "total_rules": len(self.rules),
            "categories": category_stats,
            "severity_distribution": severity_stats,
            "total_violations": total_violations,
            "total_triggers": total_triggers,
            "initialized": self._initialized,
            "index_size": len(self.keyword_index),
            "synonym_count": len(SYNONYM_DICT)
        }
    
    def validate_action(self, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """验证动作是否违反常识规则"""
        if context is None:
            context = {}
        
        ctx = context.copy()
        if "action" in ctx and ctx["action"] != action:
            ctx["action_text"] = action
        else:
            ctx["action"] = action
        
        violations = self.evaluate_context(ctx)
        
        if violations:
            highest_severity = max(v["severity"] for v in violations)
            return {
                "safe": False,
                "violations": violations,
                "highest_severity": highest_severity,
                "suggestion": self._generate_suggestion(violations)
            }
        
        return {
            "safe": True,
            "violations": [],
            "reason": "未违反任何常识规则"
        }
    
    def _generate_suggestion(self, violations: List[Dict[str, Any]]) -> str:
        """根据违规情况生成建议"""
        suggestions = []
        
        for violation in violations:
            if violation["severity"] == "high":
                suggestions.append(f"禁止此操作，原因：{violation['consequent']}")
            elif violation["severity"] == "medium":
                suggestions.append(f"建议避免，可能导致：{violation['consequent']}")
            else:
                suggestions.append(f"注意：{violation['consequent']}")
        
        return "; ".join(suggestions)
    
    def add_dynamic_rule(self, antecedent: List[str], consequent: str,
                        confidence: float = 0.8, category: str = "custom",
                        severity: str = "medium", description: str = "",
                        tags: List[str] = None) -> Dict[str, Any]:
        """动态添加新规则"""
        rule_id = f"rule_dynamic_{len(self.rules) + 1}"
        
        try:
            rule_kwargs = {
                "rule_id": rule_id,
                "antecedent": antecedent,
                "consequent": consequent,
                "confidence": confidence,
                "category": category,
                "severity": severity,
                "description": description
            }
            if tags:
                rule_kwargs["tags"] = tags
            
            rule = CommonsenseRule(**rule_kwargs)
            self.add_rule(rule)
            
            return {
                "success": True,
                "rule_id": rule_id,
                "message": "规则添加成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "规则添加失败"
            }
    
    def remove_rule(self, rule_id: str) -> Dict[str, Any]:
        """移除规则"""
        if rule_id not in self.rules:
            return {"success": False, "error": "规则不存在"}
        
        rule = self.rules[rule_id]
        
        if rule.category in self.rule_categories:
            self.rule_categories[rule.category] = [
                rid for rid in self.rule_categories[rule.category]
                if rid != rule_id
            ]
        
        del self.rules[rule_id]
        
        if rule_id in self.kg.nodes:
            del self.kg.nodes[rule_id]
        
        if self._initialized:
            self._build_indexes()
        
        return {"success": True, "message": "规则移除成功"}
    
    def add_synonym(self, keyword: str, synonyms: List[str]) -> Dict[str, Any]:
        """添加同义词"""
        if keyword in SYNONYM_DICT:
            SYNONYM_DICT[keyword].extend(synonyms)
        else:
            SYNONYM_DICT[keyword] = synonyms
        
        if self._initialized:
            self._build_indexes()
        
        return {"success": True, "message": f"同义词添加成功: {keyword} -> {synonyms}"}


def create_default_commonsense_kb(kg: EnhancedKnowledgeGraph = None) -> CommonsenseKnowledgeBase:
    """创建并初始化默认常识知识库"""
    kb = CommonsenseKnowledgeBase(kg)
    kb.initialize()
    return kb
