"""
化学学科规则集

包含12+核心化学规则：
- 原子结构：原子序数、电子排布
- 化学键：离子键、共价键、金属键
- 化学反应：化学计量、摩尔定律、质量守恒
- 溶液：浓度计算、酸碱反应
"""

from typing import List

from .disciplinary_rule import (
    DisciplinaryRule, Discipline, RuleType, RuleDifficulty,
    RuleVariable, DerivationStep
)


class ChemistryRules:
    """化学学科规则实现"""

    def __init__(self):
        self._rules = self._create_all_rules()

    def get_all_rules(self) -> List[DisciplinaryRule]:
        return self._rules

    def _create_all_rules(self) -> List[DisciplinaryRule]:
        rules = []

        # ========== 原子结构 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_atomic_number",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="原子序数定义",
            formula="Z = 质子数 = 电子数",
            description="原子序数等于原子核中的质子数，等于中性原子的电子数",
            variables=[
                RuleVariable(name="Z", symbol="Z", unit="", description="原子序数"),
                RuleVariable(name="protons", symbol="质子数", unit="", description="质子数"),
                RuleVariable(name="electrons", symbol="电子数", unit="", description="电子数"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "氢原子Z=1",
                "碳原子Z=6",
                "氧原子Z=8"
            ],
            related_concepts=["原子", "质子", "电子", "元素"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_mass_number",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="质量数定义",
            formula="A = 质子数 + 中子数",
            description="原子的质量数等于质子数加中子数",
            variables=[
                RuleVariable(name="A", symbol="A", unit="", description="质量数"),
                RuleVariable(name="protons", symbol="质子数", unit="", description="质子数"),
                RuleVariable(name="neutrons", symbol="中子数", unit="", description="中子数"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "碳-12: A=12",
                "氧-16: A=16",
                "铀-235: A=235"
            ],
            related_concepts=["质量数", "质子", "中子", "同位素"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_electron_config",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="电子排布规则",
            formula="1s² 2s² 2p⁶ 3s²...",
            description="电子按能量最低原理填充轨道，遵循泡利不相容原理和洪特规则",
            variables=[
                RuleVariable(name="n", symbol="n", unit="", description="主量子数"),
                RuleVariable(name="l", symbol="l", unit="", description="角量子数"),
                RuleVariable(name="electrons", symbol="电子数", unit="", description="电子数"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "H: 1s¹",
                "C: 1s²2s²2p²",
                "O: 1s²2s²2p⁴"
            ],
            related_concepts=["电子排布", "轨道", "量子数"],
        ))

        # ========== 化学键 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_ionic_bond",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="离子键形成",
            formula="金属 + 非金属 → 离子化合物",
            description="金属原子失去电子形成阳离子，非金属原子得到电子形成阴离子，正负离子通过静电作用结合",
            variables=[
                RuleVariable(name="metal", symbol="金属", unit="", description="金属元素"),
                RuleVariable(name="nonmetal", symbol="非金属", unit="", description="非金属元素"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "NaCl: Na+ + Cl-",
                "MgO: Mg²+ + O²-",
                "CaCl₂: Ca²+ + 2Cl-"
            ],
            related_concepts=["离子键", "阳离子", "阴离子", "静电作用"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_covalent_bond",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="共价键形成",
            formula="非金属 + 非金属 → 共价化合物",
            description="两个非金属原子通过共用电子对形成共价键",
            variables=[
                RuleVariable(name="atom1", symbol="原子1", unit="", description="非金属原子1"),
                RuleVariable(name="atom2", symbol="原子2", unit="", description="非金属原子2"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "H₂: H-H",
                "H₂O: H-O-H",
                "CO₂: O=C=O"
            ],
            related_concepts=["共价键", "共用电子对", "分子"],
        ))

        # ========== 化学反应 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_mole",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="摩尔定义",
            formula="1 mol = 6.022 × 10²³ 粒子",
            description="1摩尔任何物质所含的粒子数等于阿伏伽德罗常数",
            variables=[
                RuleVariable(name="N", symbol="N", unit="", description="粒子数"),
                RuleVariable(name="n", symbol="n", unit="mol", description="物质的量"),
                RuleVariable(name="NA", symbol="NA", unit="mol⁻¹", description="阿伏伽德罗常数"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "1mol H₂O含6.022×10²³个分子",
                "1mol C含12g",
                "1mol NaCl含6.022×10²³个离子对"
            ],
            related_concepts=["摩尔", "阿伏伽德罗常数", "物质的量"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_molar_mass",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.FORMULA,
            name="摩尔质量",
            formula="M = m / n",
            description="摩尔质量等于物质的质量除以物质的量",
            variables=[
                RuleVariable(name="M", symbol="M", unit="g/mol", description="摩尔质量"),
                RuleVariable(name="m", symbol="m", unit="g", description="质量"),
                RuleVariable(name="n", symbol="n", unit="mol", description="物质的量"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "H₂O的M=18g/mol",
                "NaCl的M=58.5g/mol",
                "CO₂的M=44g/mol"
            ],
            related_concepts=["摩尔质量", "质量", "物质的量"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_mass_conservation",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.LAW,
            name="质量守恒定律",
            formula="反应物总质量 = 生成物总质量",
            description="化学反应前后，物质的总质量保持不变",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "燃烧木材",
                "铁生锈",
                "光合作用"
            ],
            related_concepts=["质量守恒", "化学反应", "原子守恒"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_stoichiometry",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="化学计量",
            formula="n(A)/n(B) = 系数比",
            description="化学反应中各物质的物质的量之比等于化学方程式中的系数比",
            variables=[
                RuleVariable(name="n_A", symbol="n(A)", unit="mol", description="物质A的量"),
                RuleVariable(name="n_B", symbol="n(B)", unit="mol", description="物质B的量"),
                RuleVariable(name="ratio", symbol="系数比", unit="", description="系数比"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "2H₂ + O₂ → 2H₂O",
                "N₂ + 3H₂ → 2NH₃",
                "CaCO₃ → CaO + CO₂"
            ],
            related_concepts=["化学计量", "摩尔比", "反应方程式"],
        ))

        # ========== 溶液 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_concentration",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.FORMULA,
            name="物质的量浓度",
            formula="c = n / V",
            description="物质的量浓度等于溶质的物质的量除以溶液体积",
            variables=[
                RuleVariable(name="c", symbol="c", unit="mol/L", description="浓度"),
                RuleVariable(name="n", symbol="n", unit="mol", description="溶质的量"),
                RuleVariable(name="V", symbol="V", unit="L", description="溶液体积"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "生理盐水0.9% NaCl",
                "浓硫酸18mol/L",
                "pH=7的缓冲溶液"
            ],
            related_concepts=["浓度", "溶质", "溶液"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_ph",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.FORMULA,
            name="pH值计算",
            formula="pH = -log[H⁺]",
            description="pH值等于氢离子浓度的负对数",
            variables=[
                RuleVariable(name="pH", symbol="pH", unit="", description="pH值"),
                RuleVariable(name="H", symbol="[H⁺]", unit="mol/L", description="氢离子浓度"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "纯水pH=7",
                "盐酸pH=1",
                "氢氧化钠pH=13"
            ],
            related_concepts=["pH", "氢离子", "酸碱"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_acid_base",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.LAW,
            name="酸碱中和反应",
            formula="酸 + 碱 → 盐 + 水",
            description="酸和碱反应生成盐和水，H⁺ + OH⁻ → H₂O",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "HCl + NaOH → NaCl + H₂O",
                "H₂SO₄ + 2NaOH → Na₂SO₄ + 2H₂O",
                "CH₃COOH + NH₃·H₂O → CH₃COONH₄ + H₂O"
            ],
            related_concepts=["酸碱中和", "盐", "水"],
        ))

        # ========== 元素周期表 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_periodic_trend",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.LAW,
            name="元素周期律",
            formula="同周期：原子半径递减",
            description="同一周期元素，从左到右原子半径逐渐减小，金属性减弱，非金属性增强",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "Na → Mg → Al → Si → P → S → Cl",
                "金属活动性顺序",
                "元素性质递变"
            ],
            related_concepts=["周期律", "原子半径", "金属性", "非金属性"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_valence",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="化合价规则",
            formula="化合物中各元素化合价代数和为零",
            description="化合物中各元素的化合价代数和等于零",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "H₂O: H(+1)×2 + O(-2) = 0",
                "NaCl: Na(+1) + Cl(-1) = 0",
                "H₂SO₄: H(+1)×2 + S(+6) + O(-2)×4 = 0"
            ],
            related_concepts=["化合价", "氧化态", "化合物"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_octet_rule",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="八隅体规则",
            formula="最外层8电子稳定结构",
            description="原子通过得失电子或共用电子对，使最外层达到8电子稳定结构",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "Na⁺: 8电子稳定",
                "Cl⁻: 8电子稳定",
                "Ne: 天然8电子稳定"
            ],
            related_concepts=["八隅体", "稳定结构", "电子"],
        ))

        # ========== 化学平衡 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_equilibrium_constant",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.FORMULA,
            name="化学平衡常数",
            formula="K = [C]^c * [D]^d / [A]^a * [B]^b",
            description="平衡时各物质浓度以其化学计量数为指数的乘积之比",
            variables=[
                RuleVariable(name="K", symbol="K", unit="", description="平衡常数"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "H₂ + I₂ ⇌ 2HI",
                "N₂ + 3H₂ ⇌ 2NH₃",
                "合成氨反应"
            ],
            related_concepts=["化学平衡", "平衡常数", "可逆反应"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_le_chatelier",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.LAW,
            name="勒夏特列原理",
            formula="平衡移动抵消改变",
            description="改变平衡条件时，平衡向减弱该改变的方向移动",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "加压使平衡向体积减小方向移动",
                "升温使平衡向吸热方向移动",
                "增加反应物浓度使平衡正向移动"
            ],
            related_concepts=["化学平衡", "平衡移动", "条件影响"],
        ))

        # ========== 电化学 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_redox",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.PRINCIPLE,
            name="氧化还原反应",
            formula="还原剂失电子 → 氧化剂得电子",
            description="氧化还原反应中，还原剂失去电子被氧化，氧化剂得到电子被还原",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "Fe + Cu²⁺ → Fe²⁺ + Cu",
                "2Na + Cl₂ → 2NaCl",
                "电池反应"
            ],
            related_concepts=["氧化", "还原", "电子转移"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_nernst",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.FORMULA,
            name="能斯特方程",
            formula="E = E° - (RT/nF)lnQ",
            description="非标准状态下电极电势的计算公式",
            variables=[
                RuleVariable(name="E", symbol="E", unit="V", description="电极电势"),
                RuleVariable(name="E0", symbol="E°", unit="V", description="标准电极电势"),
                RuleVariable(name="R", symbol="R", unit="", description="气体常数"),
                RuleVariable(name="T", symbol="T", unit="K", description="温度"),
                RuleVariable(name="n", symbol="n", unit="", description="电子数"),
                RuleVariable(name="F", symbol="F", unit="", description="法拉第常数"),
            ],
            difficulty=RuleDifficulty.SPECIALIZED,
            confidence=0.98,
            real_world_examples=[
                "电池电压计算",
                "腐蚀电位",
                "电镀"
            ],
            related_concepts=["电极电势", "能斯特", "电化学"],
        ))

        # ========== 有机化学基础 ==========

        rules.append(DisciplinaryRule(
            rule_id="chem_hydrocarbon",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="烃类定义",
            formula="仅含C和H的有机物",
            description="烃是仅由碳和氢两种元素组成的有机化合物",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "甲烷 CH₄",
                "乙烷 C₂H₆",
                "苯 C₆H₆"
            ],
            related_concepts=["烃", "有机物", "碳氢"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_functional_group",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="官能团",
            formula="决定化学性质的原子团",
            description="官能团是决定有机物化学特性的原子或原子团",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "-OH 羟基（醇）",
                "-COOH 羧基（酸）",
                "-NH₂ 氨基（胺）"
            ],
            related_concepts=["官能团", "有机物", "化学性质"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chem_isomer",
            discipline=Discipline.CHEMISTRY,
            rule_type=RuleType.DEFINITION,
            name="同分异构体",
            formula="分子式相同，结构不同",
            description="分子式相同但结构不同的化合物互为同分异构体",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "丁烷：正丁烷和异丁烷",
                "C₂H₆O：乙醇和甲醚",
                "葡萄糖和果糖"
            ],
            related_concepts=["同分异构", "结构", "分子式"],
        ))

        return rules
