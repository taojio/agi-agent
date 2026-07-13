"""
数学学科规则集

包含15+核心数学规则：
- 代数：一元二次方程求根公式、等差数列、等比数列
- 几何：勾股定理、圆的面积/周长、三角形面积
- 微积分：导数定义、积分近似、极限计算
- 概率统计：排列组合、概率公式、标准差
"""

from typing import List

from .disciplinary_rule import (
    DisciplinaryRule, Discipline, RuleType, RuleDifficulty,
    RuleVariable, DerivationStep
)


class MathRules:
    """数学学科规则实现"""

    def __init__(self):
        self._rules = self._create_all_rules()

    def get_all_rules(self) -> List[DisciplinaryRule]:
        return self._rules

    def _create_all_rules(self) -> List[DisciplinaryRule]:
        rules = []

        # ========== 代数基础 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_quadratic_formula",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="一元二次方程求根公式",
            formula="x = (-b ± sqrt(b²-4ac)) / 2a",
            description="对于方程 ax² + bx + c = 0，其根为 x = (-b ± √(b²-4ac)) / 2a",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="二次项系数"),
                RuleVariable(name="b", symbol="b", unit="", description="一次项系数"),
                RuleVariable(name="c", symbol="c", unit="", description="常数项"),
            ],
            conditions=["a ≠ 0", "判别式 >= 0"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "计算抛体运动落地时间",
                "求解矩形面积最大化问题",
                "电路中电阻计算"
            ],
            related_concepts=["方程", "根", "判别式", "代数"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_arithmetic_sequence",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="等差数列通项公式",
            formula="an = a1 + (n-1)d",
            description="等差数列第n项等于首项加上(n-1)倍公差",
            variables=[
                RuleVariable(name="an", symbol="an", unit="", description="第n项"),
                RuleVariable(name="a1", symbol="a1", unit="", description="首项"),
                RuleVariable(name="n", symbol="n", unit="", description="项数"),
                RuleVariable(name="d", symbol="d", unit="", description="公差"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "日历日期计算",
                "楼层台阶数",
                "分期付款"
            ],
            related_concepts=["数列", "公差", "通项"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_geometric_sequence",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="等比数列通项公式",
            formula="an = a1 * r^(n-1)",
            description="等比数列第n项等于首项乘以公比的(n-1)次方",
            variables=[
                RuleVariable(name="an", symbol="an", unit="", description="第n项"),
                RuleVariable(name="a1", symbol="a1", unit="", description="首项"),
                RuleVariable(name="r", symbol="r", unit="", description="公比"),
                RuleVariable(name="n", symbol="n", unit="", description="项数"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "复利计算",
                "人口增长",
                "细菌繁殖"
            ],
            related_concepts=["数列", "公比", "指数增长"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_logarithm",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="对数公式",
            formula="log_b(a) = ln(a)/ln(b)",
            description="换底公式：以b为底a的对数等于ln(a)除以ln(b)",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="真数"),
                RuleVariable(name="b", symbol="b", unit="", description="底数"),
            ],
            conditions=["a > 0", "b > 0", "b ≠ 1"],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "pH值计算",
                "地震震级",
                "分贝计算"
            ],
            related_concepts=["对数", "指数", "换底"],
        ))

        # ========== 几何 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_pythagorean",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.THEOREM,
            name="勾股定理",
            formula="a² + b² = c²",
            description="直角三角形两直角边的平方和等于斜边的平方",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="直角边a"),
                RuleVariable(name="b", symbol="b", unit="", description="直角边b"),
                RuleVariable(name="c", symbol="c", unit="", description="斜边"),
            ],
            conditions=["直角三角形"],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "测量山高",
                "导航定位",
                "建筑设计"
            ],
            related_concepts=["直角三角形", "斜边", "三角"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_circle_area",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="圆的面积",
            formula="S = πr²",
            description="圆的面积等于圆周率乘以半径的平方",
            variables=[
                RuleVariable(name="S", symbol="S", unit="", description="面积"),
                RuleVariable(name="r", symbol="r", unit="", description="半径"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "计算圆形场地面积",
                "轮胎设计",
                "雷达覆盖范围"
            ],
            related_concepts=["圆", "面积", "圆周率"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_circle_circumference",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="圆的周长",
            formula="C = 2πr",
            description="圆的周长等于2乘以圆周率乘以半径",
            variables=[
                RuleVariable(name="C", symbol="C", unit="", description="周长"),
                RuleVariable(name="r", symbol="r", unit="", description="半径"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "计算圆形跑道长度",
                "腰带长度计算",
                "齿轮设计"
            ],
            related_concepts=["圆", "周长", "圆周率"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_triangle_area",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="三角形面积",
            formula="S = 0.5 * base * height",
            description="三角形面积等于底乘以高的一半",
            variables=[
                RuleVariable(name="S", symbol="S", unit="", description="面积"),
                RuleVariable(name="base", symbol="base", unit="", description="底边"),
                RuleVariable(name="height", symbol="height", unit="", description="高"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "计算土地面积",
                "屋顶面积",
                "三角架设计"
            ],
            related_concepts=["三角形", "面积", "底高"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_cosine_law",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.THEOREM,
            name="余弦定理",
            formula="c² = a² + b² - 2ab * cos(C)",
            description="三角形任意一边的平方等于另外两边平方和减去这两边与夹角余弦的乘积的两倍",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="边a"),
                RuleVariable(name="b", symbol="b", unit="", description="边b"),
                RuleVariable(name="c", symbol="c", unit="", description="边c"),
                RuleVariable(name="C", symbol="C", unit="", description="角C"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "测量距离",
                "导航计算",
                "三角形求解"
            ],
            related_concepts=["三角形", "余弦", "边长"],
        ))

        # ========== 微积分 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_derivative",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="导数定义",
            formula="f'(x) = lim(h→0) [f(x+h)-f(x)]/h",
            description="函数在某点的导数等于函数增量与自变量增量比值的极限",
            variables=[
                RuleVariable(name="f", symbol="f", unit="", description="函数"),
                RuleVariable(name="x", symbol="x", unit="", description="自变量"),
                RuleVariable(name="h", symbol="h", unit="", description="增量"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "速度计算",
                "切线斜率",
                "优化问题"
            ],
            related_concepts=["导数", "极限", "变化率"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_integral",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="定积分",
            formula="∫[a,b] f(x)dx = F(b)-F(a)",
            description="函数在区间[a,b]上的定积分等于原函数在端点值的差",
            variables=[
                RuleVariable(name="f", symbol="f", unit="", description="被积函数"),
                RuleVariable(name="a", symbol="a", unit="", description="下限"),
                RuleVariable(name="b", symbol="b", unit="", description="上限"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "计算面积",
                "计算体积",
                "计算功"
            ],
            related_concepts=["积分", "原函数", "面积"],
        ))

        # ========== 概率统计 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_permutation",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="排列公式",
            formula="P(n,k) = n!/(n-k)!",
            description="从n个元素中选k个进行排列的种数",
            variables=[
                RuleVariable(name="n", symbol="n", unit="", description="总数"),
                RuleVariable(name="k", symbol="k", unit="", description="选取数"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "密码组合",
                "比赛排名",
                "座位安排"
            ],
            related_concepts=["排列", "阶乘", "组合"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_combination",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="组合公式",
            formula="C(n,k) = n!/(k!(n-k)!)",
            description="从n个元素中选k个的组合种数（不考虑顺序）",
            variables=[
                RuleVariable(name="n", symbol="n", unit="", description="总数"),
                RuleVariable(name="k", symbol="k", unit="", description="选取数"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "彩票中奖",
                "球队选拔",
                "菜单选择"
            ],
            related_concepts=["组合", "阶乘", "子集"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_probability",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="概率公式",
            formula="P(A) = |A| / |Ω|",
            description="事件A发生的概率等于A包含的样本点数除以样本空间总数",
            variables=[
                RuleVariable(name="A", symbol="A", unit="", description="事件"),
                RuleVariable(name="Omega", symbol="Ω", unit="", description="样本空间"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "掷骰子",
                "抽奖概率",
                "天气预报"
            ],
            related_concepts=["概率", "样本空间", "事件"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_standard_deviation",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="标准差",
            formula="σ = sqrt(mean((x-μ)²))",
            description="标准差等于各数据与均值差的平方的平均值的平方根",
            variables=[
                RuleVariable(name="sigma", symbol="σ", unit="", description="标准差"),
                RuleVariable(name="x", symbol="x", unit="", description="数据"),
                RuleVariable(name="mu", symbol="μ", unit="", description="均值"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "数据分析",
                "质量控制",
                "风险评估"
            ],
            related_concepts=["标准差", "方差", "数据分布"],
        ))

        # ========== 三角函数 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_sin_cos",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="正弦定理",
            formula="a/sin(A) = b/sin(B) = c/sin(C) = 2R",
            description="三角形任意一边与其对角正弦的比值相等，等于外接圆直径",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="边a"),
                RuleVariable(name="b", symbol="b", unit="", description="边b"),
                RuleVariable(name="c", symbol="c", unit="", description="边c"),
                RuleVariable(name="A", symbol="A", unit="", description="角A"),
                RuleVariable(name="B", symbol="B", unit="", description="角B"),
                RuleVariable(name="C", symbol="C", unit="", description="角C"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "测量高度",
                "导航定位",
                "三角形求解"
            ],
            related_concepts=["正弦", "三角形", "外接圆"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_pythagorean_identity",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.THEOREM,
            name="三角恒等式",
            formula="sin²(x) + cos²(x) = 1",
            description="任意角度的正弦平方加余弦平方等于1",
            variables=[
                RuleVariable(name="x", symbol="x", unit="", description="角度"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "三角函数简化",
                "波形分析",
                "信号处理"
            ],
            related_concepts=["三角函数", "正弦", "余弦", "恒等式"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_tangent",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.DEFINITION,
            name="正切定义",
            formula="tan(x) = sin(x)/cos(x)",
            description="正切等于正弦除以余弦",
            variables=[
                RuleVariable(name="x", symbol="x", unit="", description="角度"),
            ],
            conditions=["cos(x) ≠ 0"],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "斜率计算",
                "角度测量",
                "导航"
            ],
            related_concepts=["正切", "正弦", "余弦"],
        ))

        # ========== 坐标系 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_distance",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="两点距离公式",
            formula="d = sqrt((x2-x1)² + (y2-y1)²)",
            description="平面上两点间的距离等于横纵坐标差的平方和的平方根",
            variables=[
                RuleVariable(name="d", symbol="d", unit="", description="距离"),
                RuleVariable(name="x1", symbol="x1", unit="", description="点1横坐标"),
                RuleVariable(name="y1", symbol="y1", unit="", description="点1纵坐标"),
                RuleVariable(name="x2", symbol="x2", unit="", description="点2横坐标"),
                RuleVariable(name="y2", symbol="y2", unit="", description="点2纵坐标"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "GPS定位",
                "地图测量",
                "游戏碰撞检测"
            ],
            related_concepts=["距离", "坐标", "平面"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_midpoint",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="中点公式",
            formula="M = ((x1+x2)/2, (y1+y2)/2)",
            description="两点中点的坐标等于两点坐标的平均值",
            variables=[
                RuleVariable(name="M", symbol="M", unit="", description="中点"),
                RuleVariable(name="x1", symbol="x1", unit="", description="点1横坐标"),
                RuleVariable(name="y1", symbol="y1", unit="", description="点1纵坐标"),
                RuleVariable(name="x2", symbol="x2", unit="", description="点2横坐标"),
                RuleVariable(name="y2", symbol="y2", unit="", description="点2纵坐标"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "寻找中心点",
                "图像处理",
                "几何作图"
            ],
            related_concepts=["中点", "坐标", "平均"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_line_slope",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="直线斜率",
            formula="k = (y2-y1)/(x2-x1)",
            description="直线的斜率等于纵坐标差除以横坐标差",
            variables=[
                RuleVariable(name="k", symbol="k", unit="", description="斜率"),
                RuleVariable(name="x1", symbol="x1", unit="", description="点1横坐标"),
                RuleVariable(name="y1", symbol="y1", unit="", description="点1纵坐标"),
                RuleVariable(name="x2", symbol="x2", unit="", description="点2横坐标"),
                RuleVariable(name="y2", symbol="y2", unit="", description="点2纵坐标"),
            ],
            conditions=["x1 ≠ x2"],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "道路坡度",
                "图表分析",
                "线性回归"
            ],
            related_concepts=["斜率", "直线", "坐标"],
        ))

        # ========== 向量 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_vector_add",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.DEFINITION,
            name="向量加法",
            formula="a + b = (ax+bx, ay+by)",
            description="两向量相加等于对应分量相加",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="向量a"),
                RuleVariable(name="b", symbol="b", unit="", description="向量b"),
            ],
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "力的合成",
                "速度叠加",
                "位移计算"
            ],
            related_concepts=["向量", "加法", "分量"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_dot_product",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="向量点积",
            formula="a·b = ax*bx + ay*by = |a||b|cos(θ)",
            description="两向量的点积等于对应分量乘积之和，也等于模的乘积乘以夹角余弦",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="向量a"),
                RuleVariable(name="b", symbol="b", unit="", description="向量b"),
                RuleVariable(name="theta", symbol="θ", unit="", description="夹角"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "功的计算",
                "投影计算",
                "夹角计算"
            ],
            related_concepts=["点积", "向量", "夹角"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_cross_product",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="向量叉积",
            formula="a×b = |a||b|sin(θ)n",
            description="两向量的叉积大小等于模的乘积乘以夹角正弦，方向垂直于两向量所在平面",
            variables=[
                RuleVariable(name="a", symbol="a", unit="", description="向量a"),
                RuleVariable(name="b", symbol="b", unit="", description="向量b"),
                RuleVariable(name="theta", symbol="θ", unit="", description="夹角"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "力矩计算",
                "面积计算",
                "三维旋转"
            ],
            related_concepts=["叉积", "向量", "法向量"],
        ))

        # ========== 复数 ==========

        rules.append(DisciplinaryRule(
            rule_id="math_complex_def",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.DEFINITION,
            name="复数定义",
            formula="z = a + bi (i² = -1)",
            description="复数由实部和虚部组成，虚数单位i的平方等于-1",
            variables=[
                RuleVariable(name="z", symbol="z", unit="", description="复数"),
                RuleVariable(name="a", symbol="a", unit="", description="实部"),
                RuleVariable(name="b", symbol="b", unit="", description="虚部"),
                RuleVariable(name="i", symbol="i", unit="", description="虚数单位"),
            ],
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "电路分析",
                "量子力学",
                "信号处理"
            ],
            related_concepts=["复数", "实部", "虚部", "虚数"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="math_euler_formula",
            discipline=Discipline.MATHEMATICS,
            rule_type=RuleType.FORMULA,
            name="欧拉公式",
            formula="e^(iθ) = cos(θ) + i*sin(θ)",
            description="复指数函数与三角函数的关系",
            variables=[
                RuleVariable(name="e", symbol="e", unit="", description="自然常数"),
                RuleVariable(name="i", symbol="i", unit="", description="虚数单位"),
                RuleVariable(name="theta", symbol="θ", unit="", description="角度"),
            ],
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.99,
            real_world_examples=[
                "傅里叶变换",
                "量子力学",
                "信号处理"
            ],
            related_concepts=["欧拉公式", "复数", "三角函数", "指数"],
        ))

        return rules
