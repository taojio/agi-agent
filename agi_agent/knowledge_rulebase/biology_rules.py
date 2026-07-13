"""
生物学科规则集

包含12+核心生物规则：
- 细胞生物学：细胞结构、细胞膜、细胞器
- 遗传学：DNA结构、遗传密码、孟德尔定律
- 生态学：生态系统、食物链、能量流动
- 生理学：光合作用、呼吸作用、细胞分裂
"""

from typing import List

from .disciplinary_rule import (
    DisciplinaryRule, Discipline, RuleType, RuleDifficulty,
    RuleVariable, DerivationStep
)


class BiologyRules:
    """生物学科规则实现"""

    def __init__(self):
        self._rules = self._create_all_rules()

    def get_all_rules(self) -> List[DisciplinaryRule]:
        return self._rules

    def _create_all_rules(self) -> List[DisciplinaryRule]:
        rules = []

        # ========== 细胞生物学 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_cell_theory",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="细胞学说",
            formula="所有生物由细胞构成",
            description="细胞是生物体结构和功能的基本单位，新细胞由老细胞分裂产生",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "人体由约37万亿个细胞组成",
                "细菌是单细胞生物",
                "植物细胞有细胞壁"
            ],
            related_concepts=["细胞", "生物体", "细胞分裂"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_cell_membrane",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="细胞膜流动镶嵌模型",
            formula="磷脂双分子层 + 蛋白质 + 糖类",
            description="细胞膜由磷脂双分子层构成基本骨架，蛋白质镶嵌其中，糖类与蛋白质结合形成糖蛋白",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "物质跨膜运输",
                "细胞识别",
                "信号传递"
            ],
            related_concepts=["细胞膜", "磷脂", "蛋白质", "糖蛋白"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_osmosis",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="渗透作用",
            formula="水从低渗→高渗",
            description="水分子通过半透膜从低浓度溶液向高浓度溶液扩散",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "植物吸水",
                "红细胞在高渗溶液中皱缩",
                "腌菜脱水"
            ],
            related_concepts=["渗透", "半透膜", "浓度", "水势"],
        ))

        # ========== 遗传学 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_dna_structure",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.DEFINITION,
            name="DNA双螺旋结构",
            formula="A-T, C-G 配对",
            description="DNA由两条反向平行的脱氧核苷酸链组成，碱基互补配对：A与T配对，C与G配对",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "DNA复制",
                "基因表达",
                "亲子鉴定"
            ],
            related_concepts=["DNA", "双螺旋", "碱基配对", "遗传物质"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_genetic_code",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="遗传密码",
            formula="3碱基 = 1氨基酸",
            description="mRNA上每3个碱基构成一个密码子，对应一种氨基酸",
            variables=[
                RuleVariable(name="codon", symbol="密码子", unit="", description="3个碱基"),
                RuleVariable(name="amino_acid", symbol="氨基酸", unit="", description="氨基酸"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "蛋白质合成",
                "基因突变",
                "基因工程"
            ],
            related_concepts=["密码子", "氨基酸", "蛋白质", "翻译"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_mendel_law_1",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="孟德尔第一定律（分离定律）",
            formula="等位基因分离",
            description="在形成配子时，成对的遗传因子彼此分离，分别进入不同的配子",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "豌豆杂交实验",
                "血型遗传",
                "遗传病概率计算"
            ],
            related_concepts=["孟德尔定律", "等位基因", "配子", "遗传"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_mendel_law_2",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="孟德尔第二定律（自由组合定律）",
            formula="非等位基因自由组合",
            description="位于非同源染色体上的非等位基因在形成配子时自由组合",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "豌豆两对性状遗传",
                "多基因遗传",
                "遗传多样性"
            ],
            related_concepts=["自由组合", "非等位基因", "同源染色体"],
        ))

        # ========== 生理学 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_photosynthesis",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.FORMULA,
            name="光合作用",
            formula="6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂",
            description="植物利用光能将二氧化碳和水转化为葡萄糖和氧气",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "植物生长",
                "氧气产生",
                "碳循环"
            ],
            related_concepts=["光合作用", "叶绿素", "光能", "葡萄糖"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_cellular_respiration",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.FORMULA,
            name="细胞呼吸",
            formula="C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP",
            description="细胞将葡萄糖氧化分解，释放能量生成ATP",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "运动时呼吸加快",
                "能量代谢",
                "发酵"
            ],
            related_concepts=["细胞呼吸", "ATP", "能量", "氧化"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_cell_division",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="细胞分裂",
            formula="间期 → 分裂期（前→中→后→末）",
            description="细胞周期包括间期和分裂期，分裂期分为前期、中期、后期、末期",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "生长发育",
                "伤口愈合",
                "癌细胞分裂"
            ],
            related_concepts=["细胞周期", "有丝分裂", "减数分裂", "染色体"],
        ))

        # ========== 生态学 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_food_chain",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="食物链",
            formula="生产者 → 初级消费者 → 次级消费者",
            description="能量沿食物链从生产者流向消费者，每经过一个营养级能量损失约90%",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "草→兔→狼",
                "浮游植物→小鱼→大鱼",
                "藻类→虾→鲸"
            ],
            related_concepts=["食物链", "营养级", "能量流动", "生态系统"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_energy_flow",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="能量流动定律",
            formula="10%定律",
            description="能量在相邻营养级间的传递效率约为10%，其余能量以热能形式散失",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "金字塔结构",
                "生态系统稳定性",
                "生物量"
            ],
            related_concepts=["能量流动", "10%定律", "营养级", "生态效率"],
        ))

        # ========== 酶与代谢 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_enzyme",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="酶的特性",
            formula="高效性、专一性、温和条件",
            description="酶是生物催化剂，具有高效性、专一性，需要在适宜温度和pH条件下发挥作用",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "唾液淀粉酶分解淀粉",
                "胃蛋白酶分解蛋白质",
                "过氧化氢酶分解过氧化氢"
            ],
            related_concepts=["酶", "催化", "代谢", "专一性"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_enzyme_activation",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="酶活性调节",
            formula="温度、pH、底物浓度影响活性",
            description="酶活性受温度、pH值、底物浓度、抑制剂等因素影响",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "高温使酶失活",
                "pH变化影响酶活性",
                "竞争性抑制"
            ],
            related_concepts=["酶活性", "最适温度", "最适pH"],
        ))

        # ========== 蛋白质与核酸 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_protein_structure",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.DEFINITION,
            name="蛋白质结构",
            formula="氨基酸 → 多肽 → 蛋白质",
            description="蛋白质由氨基酸通过肽键连接形成多肽，再折叠形成空间结构",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "胰岛素",
                "血红蛋白",
                "抗体"
            ],
            related_concepts=["蛋白质", "氨基酸", "肽键", "空间结构"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_central_dogma",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="中心法则",
            formula="DNA → RNA → 蛋白质",
            description="遗传信息从DNA流向RNA，再流向蛋白质",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "基因表达",
                "蛋白质合成",
                "遗传信息传递"
            ],
            related_concepts=["中心法则", "DNA", "RNA", "蛋白质"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_transcription",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="转录",
            formula="DNA → mRNA",
            description="以DNA为模板合成mRNA的过程",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "基因转录",
                "RNA聚合酶",
                "mRNA合成"
            ],
            related_concepts=["转录", "mRNA", "模板"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_translation",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="翻译",
            formula="mRNA → 蛋白质",
            description="以mRNA为模板，在核糖体上合成蛋白质的过程",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "核糖体合成蛋白质",
                "tRNA转运氨基酸",
                "多肽链形成"
            ],
            related_concepts=["翻译", "核糖体", "tRNA"],
        ))

        # ========== 细胞器功能 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_organelle_mitochondria",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.DEFINITION,
            name="线粒体功能",
            formula="线粒体 = 有氧呼吸场所",
            description="线粒体是细胞的能量工厂，进行有氧呼吸产生ATP",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "肌肉细胞大量线粒体",
                "心肌细胞能量供应",
                "细胞呼吸"
            ],
            related_concepts=["线粒体", "有氧呼吸", "ATP"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_organelle_chloroplast",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.DEFINITION,
            name="叶绿体功能",
            formula="叶绿体 = 光合作用场所",
            description="叶绿体含有叶绿素，是光合作用的场所",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "植物叶肉细胞",
                "光合作用",
                "氧气产生"
            ],
            related_concepts=["叶绿体", "光合作用", "叶绿素"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_organelle_ribosome",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.DEFINITION,
            name="核糖体功能",
            formula="核糖体 = 蛋白质合成场所",
            description="核糖体是蛋白质合成的场所，由rRNA和蛋白质组成",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "蛋白质合成",
                "氨基酸连接",
                "多肽链形成"
            ],
            related_concepts=["核糖体", "蛋白质合成", "rRNA"],
        ))

        # ========== 免疫与调节 ==========

        rules.append(DisciplinaryRule(
            rule_id="bio_immune",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.PRINCIPLE,
            name="免疫反应",
            formula="抗原 → 抗体",
            description="免疫系统识别抗原，产生抗体进行防御",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "疫苗接种",
                "抗体产生",
                "免疫记忆"
            ],
            related_concepts=["免疫", "抗原", "抗体", "淋巴细胞"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="bio_homeostasis",
            discipline=Discipline.BIOLOGY,
            rule_type=RuleType.LAW,
            name="稳态调节",
            formula="负反馈维持稳态",
            description="机体通过负反馈调节维持内环境稳态",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "血糖调节",
                "体温调节",
                "水盐平衡"
            ],
            related_concepts=["稳态", "负反馈", "内环境"],
        ))

        return rules
