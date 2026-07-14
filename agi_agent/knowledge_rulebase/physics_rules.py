"""
物理学科规则集

包含10+核心物理定律、公式及常见现象规则：
- 力学：牛顿三大定律、万有引力定律
- 能量：动能定理、势能公式、能量守恒定律
- 热力学：热力学第一定律、理想气体定律
- 电磁学：库仑定律、欧姆定律
- 波动：波速公式
"""

from typing import List

from .disciplinary_rule import (
    DisciplinaryRule, Discipline, RuleType, RuleDifficulty,
    RuleVariable, DerivationStep
)


class PhysicsRules:
    """物理学科规则实现"""

    def __init__(self):
        self._rules = self._create_all_rules()

    def get_all_rules(self) -> List[DisciplinaryRule]:
        """获取所有物理规则"""
        return self._rules

    def _create_all_rules(self) -> List[DisciplinaryRule]:
        rules = []

        # ========== 力学基础 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_newton_1",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="牛顿第一定律（惯性定律）",
            formula="F=0 → v=常数",
            description="物体在没有外力作用时，保持静止或匀速直线运动状态",
            variables=[
                RuleVariable(name="force", symbol="F", unit="N", description="合外力"),
                RuleVariable(name="velocity", symbol="v", unit="m/s", description="速度"),
            ],
            units={"F": "N", "v": "m/s"},
            conditions=["低速运动", "无外力作用"],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "冰面上滑行的物体",
                "太空中漂浮的宇航员",
                "匀速行驶的汽车"
            ],
            related_concepts=["惯性", "力", "运动", "速度"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_newton_2",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="牛顿第二定律",
            formula="F=ma",
            description="物体的加速度与所受合外力成正比，与质量成反比",
            variables=[
                RuleVariable(name="force", symbol="F", unit="N", description="合外力"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="acceleration", symbol="a", unit="m/s²", description="加速度"),
            ],
            units={"F": "N", "m": "kg", "a": "m/s²"},
            conditions=["低速运动", "宏观物体"],
            prerequisite_rules=["physics_newton_1"],
            dependent_rules=["physics_newton_3", "physics_kinetic_energy"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "推一辆小车",
                "火箭发射",
                "汽车加速"
            ],
            related_concepts=["力", "质量", "加速度", "动量"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_newton_3",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="牛顿第三定律",
            formula="F_ab = -F_ba",
            description="作用力与反作用力大小相等、方向相反",
            variables=[
                RuleVariable(name="force_ab", symbol="F_ab", unit="N", description="物体A对B的力"),
                RuleVariable(name="force_ba", symbol="F_ba", unit="N", description="物体B对A的力"),
            ],
            units={"F_ab": "N", "F_ba": "N"},
            prerequisite_rules=["physics_newton_2"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "划船时桨推水，水推桨",
                "走路时脚蹬地，地蹬脚",
                "火箭向下喷气，气推火箭向上"
            ],
            related_concepts=["作用力", "反作用力", "动量守恒"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_gravity",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="万有引力定律",
            formula="F=G*m1*m2/r²",
            description="任意两个质点之间存在相互吸引的万有引力",
            variables=[
                RuleVariable(name="force", symbol="F", unit="N", description="引力"),
                RuleVariable(name="mass1", symbol="m1", unit="kg", description="物体1质量"),
                RuleVariable(name="mass2", symbol="m2", unit="kg", description="物体2质量"),
                RuleVariable(name="distance", symbol="r", unit="m", description="距离"),
                RuleVariable(name="constant", symbol="G", unit="N·m²/kg²", description="万有引力常数"),
            ],
            units={"F": "N", "m1": "kg", "m2": "kg", "r": "m", "G": "6.674e-11"},
            conditions=["质点近似", "低速运动"],
            dependent_rules=["physics_orbital_speed"],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "地球绕太阳公转",
                "月球绕地球公转",
                "苹果落地"
            ],
            related_concepts=["引力", "天体运动", "质量", "距离"],
        ))

        # ========== 能量与功 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_work",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="功的定义",
            formula="W=F*d*cos(theta)",
            description="力对物体做的功等于力的大小、位移大小和力与位移夹角余弦的乘积",
            variables=[
                RuleVariable(name="work", symbol="W", unit="J", description="功"),
                RuleVariable(name="force", symbol="F", unit="N", description="力"),
                RuleVariable(name="distance", symbol="d", unit="m", description="位移"),
                RuleVariable(name="angle", symbol="theta", unit="rad", description="力与位移夹角"),
            ],
            units={"W": "J", "F": "N", "d": "m"},
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "搬箱子上楼",
                "拉车前进",
                "压缩弹簧"
            ],
            related_concepts=["功", "力", "能量", "位移"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_kinetic_energy",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="动能公式",
            formula="Ek=0.5*m*v²",
            description="物体的动能等于质量与速度平方乘积的一半",
            variables=[
                RuleVariable(name="kinetic_energy", symbol="Ek", unit="J", description="动能"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="velocity", symbol="v", unit="m/s", description="速度"),
            ],
            units={"Ek": "J", "m": "kg", "v": "m/s"},
            conditions=["低速运动"],
            prerequisite_rules=["physics_newton_2", "physics_work"],
            dependent_rules=["physics_energy_conservation"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "高速行驶的汽车",
                "飞行的子弹",
                "奔跑的运动员"
            ],
            related_concepts=["动能", "能量", "速度", "质量"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_gravitational_potential",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="重力势能公式",
            formula="Ep=m*g*h",
            description="物体的重力势能等于质量、重力加速度和高度的乘积",
            variables=[
                RuleVariable(name="potential_energy", symbol="Ep", unit="J", description="势能"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="gravity", symbol="g", unit="m/s²", description="重力加速度"),
                RuleVariable(name="height", symbol="h", unit="m", description="高度"),
            ],
            units={"Ep": "J", "m": "kg", "g": "9.8", "h": "m"},
            conditions=["地球表面附近", "恒定g"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "山顶的石头",
                "高处的水",
                "拉伸的弹簧"
            ],
            related_concepts=["势能", "重力", "能量", "高度"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_energy_conservation",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="能量守恒定律",
            formula="E_total = Ek + Ep = 常数",
            description="在封闭系统中，总能量保持不变，只在不同形式之间转化",
            variables=[
                RuleVariable(name="total_energy", symbol="E_total", unit="J", description="总能量"),
                RuleVariable(name="kinetic_energy", symbol="Ek", unit="J", description="动能"),
                RuleVariable(name="potential_energy", symbol="Ep", unit="J", description="势能"),
            ],
            units={"E_total": "J", "Ek": "J", "Ep": "J"},
            conditions=["封闭系统", "无外力做功"],
            prerequisite_rules=["physics_kinetic_energy", "physics_gravitational_potential"],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "小球从高处下落",
                "过山车",
                "水力发电"
            ],
            related_concepts=["能量守恒", "动能", "势能", "能量转化"],
        ))

        # ========== 热力学 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_thermodynamics_1",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="热力学第一定律",
            formula="Q = W + ΔU",
            description="热量等于外界对系统做的功加上系统内能的变化",
            variables=[
                RuleVariable(name="heat", symbol="Q", unit="J", description="热量"),
                RuleVariable(name="work", symbol="W", unit="J", description="功"),
                RuleVariable(name="internal_energy", symbol="ΔU", unit="J", description="内能变化"),
            ],
            units={"Q": "J", "W": "J", "ΔU": "J"},
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "加热气体使其膨胀做功",
                "冰箱制冷",
                "汽车发动机"
            ],
            related_concepts=["热力学", "内能", "热量", "功"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_ideal_gas",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="理想气体状态方程",
            formula="PV=nRT",
            description="理想气体的压强、体积、物质的量和温度之间的关系",
            variables=[
                RuleVariable(name="pressure", symbol="P", unit="Pa", description="压强"),
                RuleVariable(name="volume", symbol="V", unit="m³", description="体积"),
                RuleVariable(name="moles", symbol="n", unit="mol", description="物质的量"),
                RuleVariable(name="constant", symbol="R", unit="J/(mol·K)", description="气体常数"),
                RuleVariable(name="temperature", symbol="T", unit="K", description="温度"),
            ],
            units={"P": "Pa", "V": "m³", "n": "mol", "R": "8.314", "T": "K"},
            conditions=["理想气体", "平衡态"],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "热气球升空",
                "轮胎充气",
                "气缸中的气体"
            ],
            related_concepts=["理想气体", "压强", "体积", "温度"],
        ))

        # ========== 电磁学 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_coulomb",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="库仑定律",
            formula="F=k*q1*q2/r²",
            description="两个点电荷之间的静电力与电荷量乘积成正比，与距离平方成反比",
            variables=[
                RuleVariable(name="force", symbol="F", unit="N", description="静电力"),
                RuleVariable(name="constant", symbol="k", unit="N·m²/C²", description="库仑常数"),
                RuleVariable(name="charge1", symbol="q1", unit="C", description="电荷1"),
                RuleVariable(name="charge2", symbol="q2", unit="C", description="电荷2"),
                RuleVariable(name="distance", symbol="r", unit="m", description="距离"),
            ],
            units={"F": "N", "k": "8.988e9", "q1": "C", "q2": "C", "r": "m"},
            conditions=["点电荷", "真空"],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "静电吸引/排斥",
                "闪电",
                "电容器"
            ],
            related_concepts=["静电", "电荷", "力", "距离"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_ohm",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="欧姆定律",
            formula="V=I*R",
            description="通过导体的电流与电压成正比，与电阻成反比",
            variables=[
                RuleVariable(name="voltage", symbol="V", unit="V", description="电压"),
                RuleVariable(name="current", symbol="I", unit="A", description="电流"),
                RuleVariable(name="resistance", symbol="R", unit="Ω", description="电阻"),
            ],
            units={"V": "V", "I": "A", "R": "Ω"},
            conditions=["恒定温度", "线性电阻"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "家庭电路",
                "手机充电",
                "电器工作"
            ],
            related_concepts=["电流", "电压", "电阻", "电路"],
        ))

        # ========== 波动 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_wave_speed",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="波速公式",
            formula="v=f*lambda",
            description="波速等于频率与波长的乘积",
            variables=[
                RuleVariable(name="velocity", symbol="v", unit="m/s", description="波速"),
                RuleVariable(name="frequency", symbol="f", unit="Hz", description="频率"),
                RuleVariable(name="wavelength", symbol="lambda", unit="m", description="波长"),
            ],
            units={"v": "m/s", "f": "Hz", "lambda": "m"},
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "声波传播",
                "电磁波",
                "水波"
            ],
            related_concepts=["波", "频率", "波长", "速度"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_momentum",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="动量公式",
            formula="p=m*v",
            description="物体的动量等于质量与速度的乘积",
            variables=[
                RuleVariable(name="momentum", symbol="p", unit="kg·m/s", description="动量"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="velocity", symbol="v", unit="m/s", description="速度"),
            ],
            units={"p": "kg·m/s", "m": "kg", "v": "m/s"},
            prerequisite_rules=["physics_newton_2"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "撞球",
                "火箭发射",
                "汽车碰撞"
            ],
            related_concepts=["动量", "质量", "速度", "冲量"],
        ))

        # ========== 简谐运动 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_shm_period",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="简谐运动周期（弹簧振子）",
            formula="T=2π*sqrt(m/k)",
            description="弹簧振子的周期等于2π乘以质量与劲度系数比值的平方根",
            variables=[
                RuleVariable(name="period", symbol="T", unit="s", description="周期"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="spring_constant", symbol="k", unit="N/m", description="劲度系数"),
            ],
            units={"T": "s", "m": "kg", "k": "N/m"},
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "弹簧振动",
                "钟摆",
                "汽车减震器"
            ],
            related_concepts=["简谐运动", "周期", "弹簧"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_pendulum",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="单摆周期",
            formula="T=2π*sqrt(L/g)",
            description="单摆的周期等于2π乘以摆长与重力加速度比值的平方根",
            variables=[
                RuleVariable(name="period", symbol="T", unit="s", description="周期"),
                RuleVariable(name="length", symbol="L", unit="m", description="摆长"),
                RuleVariable(name="gravity", symbol="g", unit="m/s²", description="重力加速度"),
            ],
            units={"T": "s", "L": "m", "g": "m/s²"},
            conditions=["小角度摆动"],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "摆钟",
                "秋千",
                "地震仪"
            ],
            related_concepts=["单摆", "周期", "重力"],
        ))

        # ========== 光学 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_snell_law",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="折射定律（斯涅尔定律）",
            formula="n1*sin(θ1)=n2*sin(θ2)",
            description="入射角与折射角的正弦之比等于两介质折射率的反比",
            variables=[
                RuleVariable(name="n1", symbol="n1", unit="", description="介质1折射率"),
                RuleVariable(name="n2", symbol="n2", unit="", description="介质2折射率"),
                RuleVariable(name="theta1", symbol="θ1", unit="rad", description="入射角"),
                RuleVariable(name="theta2", symbol="θ2", unit="rad", description="折射角"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "水中筷子弯曲",
                "眼镜",
                "光纤通信"
            ],
            related_concepts=["折射", "折射率", "光"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_mirror",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="凸透镜成像",
            formula="1/u+1/v=1/f",
            description="物距、像距与焦距满足倒数关系",
            variables=[
                RuleVariable(name="object_distance", symbol="u", unit="m", description="物距"),
                RuleVariable(name="image_distance", symbol="v", unit="m", description="像距"),
                RuleVariable(name="focal_length", symbol="f", unit="m", description="焦距"),
            ],
            units={"u": "m", "v": "m", "f": "m"},
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "照相机",
                "眼镜",
                "显微镜"
            ],
            related_concepts=["透镜", "成像", "焦点"],
        ))

        # ========== 相对论基础 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_lorentz_factor",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="洛伦兹因子",
            formula="γ=1/sqrt(1-v²/c²)",
            description="描述相对论效应的因子，速度越接近光速，效应越显著",
            variables=[
                RuleVariable(name="gamma", symbol="γ", unit="", description="洛伦兹因子"),
                RuleVariable(name="velocity", symbol="v", unit="m/s", description="速度"),
                RuleVariable(name="light_speed", symbol="c", unit="m/s", description="光速"),
            ],
            conditions=["v < c"],
            difficulty=RuleDifficulty.SPECIALIZED,
            confidence=0.99,
            real_world_examples=[
                "粒子加速器",
                "GPS卫星时钟校准",
                "宇宙射线"
            ],
            related_concepts=["相对论", "时间膨胀", "长度收缩"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_mass_energy",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.LAW,
            name="质能方程",
            formula="E=mc²",
            description="物体的能量等于其质量乘以光速的平方",
            variables=[
                RuleVariable(name="energy", symbol="E", unit="J", description="能量"),
                RuleVariable(name="mass", symbol="m", unit="kg", description="质量"),
                RuleVariable(name="light_speed", symbol="c", unit="m/s", description="光速"),
            ],
            units={"E": "J", "m": "kg", "c": "3e8"},
            difficulty=RuleDifficulty.SPECIALIZED,
            confidence=0.99,
            real_world_examples=[
                "核裂变",
                "核聚变",
                "正负电子对湮灭"
            ],
            related_concepts=["质能守恒", "核反应", "相对论"],
        ))

        # ========== 电学扩展 ==========

        rules.append(DisciplinaryRule(
            rule_id="physics_power",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="电功率",
            formula="P=V*I",
            description="电功率等于电压乘以电流",
            variables=[
                RuleVariable(name="power", symbol="P", unit="W", description="功率"),
                RuleVariable(name="voltage", symbol="V", unit="V", description="电压"),
                RuleVariable(name="current", symbol="I", unit="A", description="电流"),
            ],
            units={"P": "W", "V": "V", "I": "A"},
            prerequisite_rules=["physics_ohm"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "电器功率",
                "电费计算",
                "节能灯"
            ],
            related_concepts=["功率", "电压", "电流", "电能"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="physics_capacitor",
            discipline=Discipline.PHYSICS,
            rule_type=RuleType.FORMULA,
            name="电容公式",
            formula="C=Q/V",
            description="电容等于电荷量除以电压",
            variables=[
                RuleVariable(name="capacitance", symbol="C", unit="F", description="电容"),
                RuleVariable(name="charge", symbol="Q", unit="C", description="电荷量"),
                RuleVariable(name="voltage", symbol="V", unit="V", description="电压"),
            ],
            units={"C": "F", "Q": "C", "V": "V"},
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "手机电池",
                "相机闪光灯",
                "计算机内存"
            ],
            related_concepts=["电容", "电荷", "电压"],
        ))

        return rules
