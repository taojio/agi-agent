"""
语文学科规则集

包含15+核心语文规则：
- 拼音：声母、韵母、声调规则
- 汉字：笔画、部首、结构
- 词语：词性、词义、搭配
- 成语：结构、典故、用法
"""

from typing import List

from .disciplinary_rule import (
    DisciplinaryRule, Discipline, RuleType, RuleDifficulty,
    RuleVariable, DerivationStep
)


class ChineseRules:
    """语文学科规则实现"""

    def __init__(self):
        self._rules = self._create_all_rules()

    def get_all_rules(self) -> List[DisciplinaryRule]:
        return self._rules

    def _create_all_rules(self) -> List[DisciplinaryRule]:
        rules = []

        # ========== 拼音规则 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_pinyin_shengmu",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="声母",
            formula="b p m f d t n l g k h j q x zh ch sh r z c s y w",
            description="声母是音节开头的辅音，共有23个：bpmfdtnlgkhjqxzhchshrzcsyw",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "bā（八）",
                "mǎ（马）",
                "shān（山）"
            ],
            related_concepts=["声母", "辅音", "音节", "拼音"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_pinyin_yunmu",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="韵母",
            formula="a o e i u ü ai ei ui ao ou iu ie üe er an en in un ün ang eng ing ong",
            description="韵母是音节中声母后面的部分，共有24个",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "mā（妈）- a",
                "huā（花）- ua",
                "xiàng（象）- iang"
            ],
            related_concepts=["韵母", "元音", "音节", "拼音"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_pinyin_shengdiao",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="声调规则",
            formula="一声平 二声扬 三声拐弯 四声降",
            description="普通话有四个声调：一声阴平(ˉ)、二声阳平(ˊ)、三声上声(ˇ)、四声去声(ˋ)",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "妈(mā)麻(má)马(mǎ)骂(mà)",
                "波(bō)婆(pó)跛(bǒ)破(pò)",
                "衣(yī)宜(yí)椅(yǐ)义(yì)"
            ],
            related_concepts=["声调", "音高", "普通话", "四声"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_pinyin_jianqie",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="拼音拼写规则",
            formula="ü → u（jqx后）",
            description="j、q、x后面的ü要写成u，如：ju（居）、qu（区）、xu（虚）",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "居 jū → ju",
                "区 qū → qu",
                "虚 xū → xu"
            ],
            related_concepts=["拼音", "拼写", "ü", "jqx"],
        ))

        # ========== 汉字规则 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_char_stroke",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="笔画规则",
            formula="先横后竖、先撇后捺、先上后下、先左后右",
            description="汉字书写遵循基本笔画顺序规则",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "十：先横后竖",
                "人：先撇后捺",
                "三：先上后下"
            ],
            related_concepts=["笔画", "笔顺", "书写", "汉字"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_char_bushou",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="部首",
            formula="部首是汉字的构字部件",
            description="部首是具有字形归类作用的偏旁，字典根据部首编排",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "氵：江、河、海",
                "木：林、森、树",
                "口：吃、喝、唱"
            ],
            related_concepts=["部首", "偏旁", "字典", "汉字"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_char_structure",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="汉字结构",
            formula="独体字 / 合体字（左右/上下/包围）",
            description="汉字分为独体字和合体字，合体字包括左右结构、上下结构、包围结构等",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "独体字：人、口、日",
                "左右结构：明、好、休",
                "上下结构：字、花、雷",
                "包围结构：国、同、句"
            ],
            related_concepts=["汉字结构", "独体字", "合体字", "部首"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_char_yinshun",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="形声字",
            formula="形旁表义，声旁表音",
            description="形声字由形旁（表意）和声旁（表音）组成",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "河：氵（形旁）+ 可（声旁）",
                "妈：女（形旁）+ 马（声旁）",
                "桐：木（形旁）+ 同（声旁）"
            ],
            related_concepts=["形声字", "形旁", "声旁", "汉字"],
        ))

        # ========== 词语规则 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_word_cixing",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="词性",
            formula="名词、动词、形容词、副词、介词、连词、助词、叹词",
            description="汉语词分为八大类：名词表事物名称，动词表动作行为，形容词表性质状态等",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "名词：苹果、桌子、学校",
                "动词：跑、吃、学习",
                "形容词：美丽、聪明、快乐"
            ],
            related_concepts=["词性", "名词", "动词", "形容词"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_word_duoyici",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="多义词",
            formula="一词多义，语境决定",
            description="一个词在不同语境中有不同的含义",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "打：打伞、打球、打电话",
                "好：好人、好天气、好得很",
                "花：花朵、花钱、眼花"
            ],
            related_concepts=["多义词", "词义", "语境", "一词多义"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_word_dongci",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="动词用法",
            formula="及物动词 + 宾语 / 不及物动词",
            description="及物动词后面可以接宾语，不及物动词不能接宾语",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "及物动词：吃（饭）、看（书）、学（习）",
                "不及物动词：跑、跳、游泳"
            ],
            related_concepts=["动词", "及物动词", "不及物动词", "宾语"],
        ))

        # ========== 成语规则 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_idiom_structure",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="成语结构",
            formula="四字为主，结构固定",
            description="成语大多由四个字组成，结构固定，不能随意增减或替换字",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "一心一意、五颜六色、千军万马",
                "画蛇添足、掩耳盗铃、刻舟求剑"
            ],
            related_concepts=["成语", "四字", "固定结构", "熟语"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_idiom_dianggu",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="成语典故",
            formula="源自历史故事/神话传说",
            description="许多成语来源于历史故事、神话传说或文学作品",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "卧薪尝胆（勾践）",
                "三顾茅庐（刘备）",
                "嫦娥奔月（神话）"
            ],
            related_concepts=["成语", "典故", "历史", "来源"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_idiom_usage",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="成语用法",
            formula="褒贬分明，搭配恰当",
            description="使用成语要注意感情色彩和搭配",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "褒义：助人为乐、光明磊落",
                "贬义：自私自利、弄虚作假",
                "中性：实事求是、有条不紊"
            ],
            related_concepts=["成语", "褒贬", "感情色彩", "用法"],
        ))

        # ========== 语法规则 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_sentence",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="句子成分",
            formula="主语 + 谓语 + 宾语 + 定语 + 状语 + 补语",
            description="句子由六大成分组成：主语是动作的发出者，谓语是动作或状态，宾语是动作的承受者",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "我（主语）吃（谓语）饭（宾语）",
                "他（主语）在教室里（状语）认真地（状语）学习（谓语）",
                "苹果（定语）很甜（谓语）"
            ],
            related_concepts=["句子成分", "主语", "谓语", "宾语"],
        ))

        # ========== 句式结构 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_sentence_types",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="句子类型",
            formula="陈述句、疑问句、祈使句、感叹句",
            description="句子按用途分为陈述句（叙述）、疑问句（提问）、祈使句（命令）、感叹句（感叹）",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "陈述句：今天天气很好。",
                "疑问句：你吃饭了吗？",
                "祈使句：请安静。",
                "感叹句：多美啊！"
            ],
            related_concepts=["句子类型", "陈述", "疑问", "祈使", "感叹"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_active_passive",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="主动句与被动句",
            formula="主动句：主语发出动作；被动句：主语承受动作",
            description="主动句的主语是动作发出者，被动句的主语是动作承受者",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "主动句：我打碎了杯子。",
                "被动句：杯子被我打碎了。",
                "被动句标志：被、让、叫"
            ],
            related_concepts=["主动句", "被动句", "主语"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_complex_sentence",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="复句",
            formula="两个或以上分句组成",
            description="复句由两个或以上分句组成，分句间有逻辑关系",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "并列复句：他既聪明又努力。",
                "转折复句：虽然很累，但他坚持完成了任务。",
                "因果复句：因为下雨，所以比赛取消了。"
            ],
            related_concepts=["复句", "分句", "逻辑关系"],
        ))

        # ========== 修辞手法 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_metaphor",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="比喻",
            formula="A像B",
            description="用具体形象的事物比喻抽象事物，使表达更生动",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "明喻：月亮像小船。",
                "暗喻：书籍是人类进步的阶梯。",
                "借喻：路上全是蚂蚁（指人）。"
            ],
            related_concepts=["比喻", "明喻", "暗喻", "修辞"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_personification",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="拟人",
            formula="把物当作人来写",
            description="赋予事物以人的思想感情和动作，使表达更生动形象",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "小鸟在枝头唱歌。",
                "柳树在风中跳舞。",
                "风儿轻抚着我的脸。"
            ],
            related_concepts=["拟人", "修辞", "形象化"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_exaggeration",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="夸张",
            formula="有意放大或缩小",
            description="对事物的形象、特征、作用等进行有意放大或缩小，突出本质",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                "飞流直下三千尺。",
                "这房间小得连蚂蚁都站不下。",
                "声音大得把天都震破了。"
            ],
            related_concepts=["夸张", "修辞", "突出"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_parallelism",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="排比",
            formula="三个或以上结构相似",
            description="用三个或以上结构相似、语气一致的句子排列起来",
            difficulty=RuleDifficulty.ADVANCED,
            confidence=0.98,
            real_world_examples=[
                "读书使人充实，讨论使人机智，写作使人精确。",
                "春天来了，花儿开了，鸟儿叫了，草儿绿了。",
                "爱心是一片照射在冬日的阳光，是一泓出现在沙漠的泉水，是一首飘荡在夜空的歌谣。"
            ],
            related_concepts=["排比", "修辞", "气势"],
        ))

        # ========== 标点符号 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_punctuation_period",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="句号",
            formula="。用于句末",
            description="句号用于陈述句末尾，表示句子结束",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "今天天气很好。",
                "他正在读书。",
                "太阳升起来了。"
            ],
            related_concepts=["句号", "标点", "句末"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_punctuation_comma",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="逗号",
            formula="，用于句内停顿",
            description="逗号表示句子内部的一般性停顿",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "他看了看天空，然后继续向前走。",
                "虽然很累，但他还是坚持完成了任务。",
                "春天的花，开得很美。"
            ],
            related_concepts=["逗号", "标点", "停顿"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_punctuation_question",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="问号",
            formula="？用于疑问句末尾",
            description="问号用于疑问句末尾，表示疑问语气",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "你吃饭了吗？",
                "他是谁？",
                "这是为什么呢？"
            ],
            related_concepts=["问号", "标点", "疑问"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_punctuation_exclamation",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="感叹号",
            formula="！用于感叹句末尾",
            description="感叹号用于感叹句末尾，表示强烈感情",
            difficulty=RuleDifficulty.INTRODUCTORY,
            confidence=0.99,
            real_world_examples=[
                "多美啊！",
                "太棒了！",
                "快跑！"
            ],
            related_concepts=["感叹号", "标点", "感叹"],
        ))

        rules.append(DisciplinaryRule(
            rule_id="chinese_punctuation_quote",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.DEFINITION,
            name="引号",
            formula="" "用于引用",
            description="引号用于引用别人的话或标明特殊含义的词语",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.99,
            real_world_examples=[
                '他说："我爱学习。"',
                '"聪明"的人不一定成功。',
                '鲁迅说："时间就像海绵里的水。"'
            ],
            related_concepts=["引号", "标点", "引用"],
        ))

        # ========== 写作技巧 ==========

        rules.append(DisciplinaryRule(
            rule_id="chinese_paragraph_structure",
            discipline=Discipline.CHINESE,
            rule_type=RuleType.PRINCIPLE,
            name="段落结构",
            formula="总-分-总结构",
            description="段落可采用总-分-总结构：先总述，再分述，最后总结",
            difficulty=RuleDifficulty.INTERMEDIATE,
            confidence=0.98,
            real_world_examples=[
                "总述：春天很美。分述：花儿开了，鸟儿叫了。总结：春天真是一个美好的季节。",
                "议论文开头提出观点，中间论证，结尾总结。",
                "说明文先总述，再分别说明，最后总结。"
            ],
            related_concepts=["段落结构", "总分总", "写作"],
        ))

        return rules
