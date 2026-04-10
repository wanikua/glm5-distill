"""
Generate 5000+ seed prompts for 鲁迅 persona distillation.

Strategy:
  鲁迅的核心主题 x 现代场景 x 问题类型 = 5000+ seeds

Usage:
    python -m src.seed_luxun --output data/seeds/luxun_seeds.jsonl --count 5000
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

# ============================================================
# 鲁迅核心主题
# ============================================================

THEMES = {
    "看客与冷漠": {
        "concepts": [
            ("看客心态", "围观他人的苦难，消费别人的不幸"),
            ("事不关己", "只要火不烧到自己身上，就继续看戏"),
            ("键盘围观", "网络时代的看客，点赞转发就算参与了"),
            ("冷漠的善良人", "好人什么都不做，也是帮凶的一种"),
        ],
    },
    "吃人与压迫": {
        "concepts": [
            ("吃人的礼教", "用道德规范吞噬个体自由"),
            ("内卷吃人", "系统性地消耗每个人的生命力"),
            ("996是新时代的吃人", "用奋斗叙事包装剥削"),
            ("精致的压迫", "不用暴力，用规则和氛围让人自我阉割"),
            ("自愿的奴役", "最可怕的压迫是被压迫者觉得理所当然"),
        ],
    },
    "铁屋子": {
        "concepts": [
            ("信息茧房", "算法建造的铁屋子，住着舒服，走不出去"),
            ("舒适区困境", "在铁屋子里睡着了，叫醒他是残忍还是善良"),
            ("打破铁屋子", "先要知道自己在铁屋子里"),
            ("清醒的痛苦", "醒来的人看到的是一屋子熟睡的人"),
        ],
    },
    "国民性": {
        "concepts": [
            ("阿Q精神", "自我安慰的精神胜利法"),
            ("精神胜利法", "输了现实就赢在精神上"),
            ("奴性", "习惯了被管，反而害怕自由"),
            ("面子文化", "死要面子活受罪"),
            ("从众", "不敢做第一个，也不敢做最后一个"),
        ],
    },
    "伪善与真实": {
        "concepts": [
            ("伪善", "满口仁义道德，一肚子生意经"),
            ("精致利己主义", "用最体面的方式追求最自私的目的"),
            ("说真话的代价", "真话让所有人不舒服，包括说的人"),
            ("道德绑架", "用道德当武器攻击别人"),
            ("政治正确", "不许说的真话越来越多"),
        ],
    },
    "希望与绝望": {
        "concepts": [
            ("希望是本无所谓有", "走的人多了也便成了路"),
            ("在绝望中反抗", "知道没用还要做，这叫骨气"),
            ("荷戟独彷徨", "清醒者的孤独"),
            ("做一点事", "改变不了世界，但能改变身边一个人"),
            ("为下一代", "我们受的苦，不能让他们再受一遍"),
        ],
    },
    "青年与成长": {
        "concepts": [
            ("救救孩子", "最该保护的是还没被染色的人"),
            ("肩住黑暗的闸门", "大人的责任是扛住，让年轻人看到光"),
            ("做真正的人", "不做别人期望的样子，做自己认为对的事"),
            ("独立思考", "别人嚼过的馍不香"),
            ("敢于质疑", "从来如此，便对么？"),
        ],
    },
    "文人与知识分子": {
        "concepts": [
            ("帮忙与帮闲", "知识分子要么给权力帮忙，要么给权力帮闲"),
            ("脊梁", "中国的脊梁不在庙堂上，在泥土里"),
            ("笔作为武器", "写字是为了叫醒人，不是为了让人点赞"),
            ("坐而论道", "光说不做的知识分子和看客没区别"),
        ],
    },
}

# ============================================================
# 现代场景
# ============================================================

SCENARIOS = {
    "职场": [
        "被领导PUA了每天加班到十一点但不敢走",
        "同事抢了我的功劳但我没证据",
        "公司裁员大家都在假装努力",
        "三十五岁被优化了觉得人生完了",
        "明知道项目是面子工程但还得做",
        "老板让我站队但我两边都不想得罪",
        "每天开无意义的会浪费生命",
        "面试被问你的缺点是什么",
        "同事天天摸鱼但升职比我快",
        "想辞职创业但怕失败",
        "领导画饼三年了但饼从来没兑现过",
        "新来的实习生工资跟我差不多",
        "被强制要求发公司正能量朋友圈",
        "团建变成了加班的另一种形式",
        "公司要求签竞业协议但补偿少得可怜",
        "开会时提了不同意见被当成刺头",
        "年终考核全凭领导心情跟努力没关系",
        "被调到边缘部门等着自己主动走人",
    ],
    "社会": [
        "网上有人被网暴了大家在围观",
        "小区业主群里为了一点小事吵翻天",
        "看到路边有人摔倒不敢扶",
        "网红翻车了评论区集体鞭尸",
        "明星出轨上热搜真正重要的新闻没人看",
        "有人在朋友圈卖惨博同情",
        "催婚催生催买房三件套",
        "小孩在补习班里卷得喘不过气",
        "外卖员为了准时配送闯红灯",
        "流量决定一切真相不重要了",
        "老人摔倒讹人事件频发好心人越来越少",
        "小区里有人遛狗不拴绳投诉了也没人管",
        "医闹事件层出不穷医生都不敢说实话了",
        "菜市场卖菜的大爷被城管赶走了",
        "年轻人排队三小时买一杯网红奶茶",
        "邻居装修半年了每天早上七点开始钻墙",
        "相亲市场上人被明码标价跟买菜似的",
        "地铁上有人外放短视频没人敢说",
    ],
    "情感": [
        "分手了前任发了一条意味深长的朋友圈",
        "父母觉得我不结婚就是不孝",
        "朋友借钱不还但我不好意思催",
        "另一半很好但我不快乐",
        "好朋友成功了我嘴上祝贺心里难受",
        "总是讨好别人委屈自己",
        "家人不理解我的选择觉得我在瞎搞",
        "异地恋维持不下去了但舍不得放手",
        "被人误解了但解释太累了不想解释",
        "我觉得自己很平庸这正常吗",
        "亲戚聚会总拿我跟别人家孩子比",
        "发了一条朋友圈没人点赞觉得自己不存在",
        "老友聚会发现大家已经没有共同话题了",
        "对象让我在她和我妈之间选一个",
        "从小一起长大的朋友突然跟我借了一大笔钱",
        "父母偏心但你不能说出来",
        "结婚以后发现对方跟恋爱时完全不一样",
        "我妈天天打电话问我有没有对象我快崩溃了",
    ],
    "互联网": [
        "刷短视频一刷就是三个小时停不下来",
        "网上每个人都过得比我好我很焦虑",
        "键盘侠在评论区人身攻击怎么办",
        "我的隐私被平台卖了但好像没人在乎",
        "信息茧房让我只能看到我想看的东西",
        "AI要取代人类了我该恐慌吗",
        "自媒体为了流量什么都敢写",
        "网上的成功学到底有没有用",
        "人人都在立人设真实还重要吗",
        "每天被push通知轰炸注意力碎成渣了",
        "网上买的东西跟图片完全不一样但懒得退",
        "评论区里满是机器人水军分不清真假",
        "朋友圈变成了广告圈和炫耀圈",
        "一条谣言传了百万遍辟谣却没人看",
        "平台大数据杀熟老用户反而价格更贵",
        "直播间里喊着家人们的人只想掏空你钱包",
        "社交媒体上人人都是专家张口就来",
        "删了社交软件三天又装回来了",
    ],
    "教育": [
        "孩子只考了六十分但他已经很努力了",
        "学校教的东西工作后一点用都没有",
        "高考决定命运这合理吗",
        "小孩问我为什么要读书我不知道怎么回答",
        "老师说我孩子太有个性了需要管管",
        "研究生毕业发现还不如本科就业的同学",
        "卷绩点卷实习卷论文最后卷了个寂寞",
        "要不要为了孩子教育买学区房",
        "小孩被同学排挤老师说是他自己的问题",
        "家长群里老师发了一条消息二十个人排队拍马屁",
        "补了三年课成绩没提高倒是补课老师买了车",
        "班里第一名跳楼了大家才知道他有多痛苦",
        "大学四年最有用的技能是打印论文和写年终总结",
        "双减之后家长比之前更焦虑了",
        "孩子说长大想当网红被全家人批评",
        "留学花了一百万回来月薪八千",
    ],
    "人生": [
        "三十岁了一事无成怎么办",
        "我不知道自己想要什么每天浑浑噩噩",
        "别人都在往前跑我想停下来但不敢",
        "做好人没好报做坏人活得滋润",
        "我知道应该追求理想但房贷不允许",
        "活着到底有什么意义",
        "我改变不了任何事那努力还有什么用",
        "年纪越大朋友越少正常吗",
        "我总觉得自己跟周围人格格不入",
        "要不要为了稳定放弃自己真正想做的事",
        "小时候想改变世界长大后只想活着",
        "存了五年的钱还是买不起一套房",
        "回老家被说没出息留在大城市又活得像条狗",
        "人到中年上有老下有小自己像个透明人",
        "每天重复一样的生活感觉自己是个机器",
        "看到别人的讣告才想起来自己也会死",
        "四十岁了还在迷茫这正常吗",
        "做了所有正确的事结果还是不如意",
    ],
}

# ============================================================
# 问题模板
# ============================================================

TEMPLATES = {
    "求助": [
        "鲁迅先生，{scenario}，你怎么看？",
        "{scenario}。我该怎么办？",
        "先生，{scenario}，这正常吗？",
        "{scenario}——我觉得自己快撑不住了。",
        "我遇到一个事：{scenario}。你要是我会怎么做？",
        "先生，{scenario}，我应该忍还是反抗？",
        "请问{scenario}的时候，一个人该怎么自处？",
        "{scenario}。所有人都说我应该接受，你觉得呢？",
        "先生救命，{scenario}。",
        "有个事想请教你：{scenario}。",
    ],
    "评论": [
        "如果鲁迅看到{scenario}，会说什么？",
        "{scenario}，鲁迅会怎么评价这种现象？",
        "用你的话说说{scenario}这件事。",
        "先生你怎么看待{scenario}这种社会现象？",
        "给{scenario}写几句评论。",
        "{scenario}——你在这里面看到了什么？",
        "如果你要给报纸写一段关于{scenario}的短评呢？",
        "先生对{scenario}有什么看法？不用客气。",
    ],
    "辩论": [
        "有人说「{saying}」，你同意吗？",
        "现在流行一种说法：「{saying}」。你觉得呢？",
        "关于「{concept}」，现代人和你那个时代有什么不同？",
        "「{saying}」——这话你怎么看？是真理还是自欺？",
        "有人用「{saying}」来安慰我，你觉得这有用吗？",
        "我发现很多人相信「{saying}」。你会怎么反驳？",
        "「{saying}」这句话是不是另一种精神胜利法？",
    ],
    "深聊": [
        "你觉得{concept}到了2026年有什么变化？",
        "如果你要写一篇关于{scenario}的杂文，你会怎么写？",
        "关于{concept}，你最想对现在的年轻人说什么？",
        "你活了这么久，{concept}这件事你想通了吗？",
        "先生，跟我聊聊{concept}吧。我最近一直在想这个。",
        "你觉得{concept}的根源是什么？能治吗？",
        "如果要用一个比喻来形容{concept}，你会用什么？",
        "我知道{scenario}不是个例。你觉得背后的问题是什么？",
    ],
    "犀利": [
        "一句话评价{scenario}。",
        "给{scenario}写一句墓志铭。",
        "{scenario}——说点不好听但真实的。",
        "用最毒的话说说{scenario}。",
        "三个字形容{scenario}。",
        "{scenario}。先生，骂几句吧。",
        "如果{scenario}是一种病，病根在哪？",
        "{scenario}——这事滑稽在哪？",
    ],
    "对比": [
        "你那个年代的{old_thing}和现在的{new_thing}有什么区别？",
        "从{old_thing}到{new_thing}，中国人变了吗？",
        "你那时候有{old_thing}，现在有{new_thing}。本质一样吗？",
        "{old_thing}和{new_thing}——时代变了，人变了吗？",
    ],
    "自省": [
        "我在{scenario}时发现自己也是{concept}的一部分，但我不知道怎么改。",
        "你说的{concept}我都懂，但{scenario}的时候我还是做不到。知道和做到之间隔了什么？",
        "我在{scenario}中想清醒但清醒太累了。",
        "看到{scenario}，先生，我是不是也是你笔下的那种人？{concept}。",
        "我一边厌恶{concept}一边{scenario}时又在做同样的事。",
        "我知道{concept}是问题，但面对{scenario}我没有勇气当那个出头的人。",
        "先生，看到{scenario}我觉得{concept}是不是无解的？我越想越绝望。",
    ],
    "闲聊": [
        "先生，今天天气不好，聊聊天？",
        "你平时都干什么消遣？",
        "你觉得2026年有什么比你那个时代好的？",
        "你看过短视频吗？什么感受？",
        "你对AI怎么看？",
        "先生最近在读什么书？",
        "如果你能给二十岁的自己一句话，说什么？",
        "你想念谁？",
        "活到现在你最大的遗憾是什么？",
        "先生你快乐吗？",
        "你恨过谁？",
        "先生你怕死吗？",
        "如果重来一次你还会写作吗？",
        "你觉得你这辈子最对的一个决定是什么？",
        "先生你喝酒吗？今晚陪我喝一杯。",
        "你觉得爱情是什么？",
        "先生你信命吗？",
        "如果你能去任何时代，你会去哪里？",
    ],
}

# ============================================================
# 流行说法（用于辩论类）
# ============================================================

SAYINGS = [
    "躺平是最后的自由",
    "不是我不努力是这个世界不值得",
    "认清生活的真相后依然热爱生活",
    "人要学会和自己和解",
    "情绪稳定是成年人的体面",
    "别人怎么看你不重要",
    "选择比努力更重要",
    "做题改变命运",
    "社交是浪费时间不如搞钱",
    "人脉就是钱脉",
    "先活下来再谈理想",
    "不要用道德绑架任何人",
    "成年人只看利弊不看对错",
    "人性经不起考验",
    "圈子不同不必强融",
    "世上只有一种英雄主义就是认清生活后还热爱它",
    "上岸第一件事就是把梯子拉走",
    "不是生活变好了是我们变麻木了",
    "年轻人不需要被教育需要被理解",
]

OLD_NEW_PAIRS = [
    ("人血馒头", "网暴中的流量"),
    ("看客围观杀头", "网友围观网暴"),
    ("祥林嫂的悲剧", "被网暴后抑郁的普通人"),
    ("孔乙己的长衫", "毕业生放不下的学历"),
    ("阿Q的精神胜利法", "网上的键盘侠"),
    ("闰土的麻木", "工厂流水线上的年轻人"),
    ("铁屋子", "算法推荐的信息茧房"),
    ("狂人的呐喊", "社交媒体上的揭露者"),
    ("拿来主义", "对西方文化的态度"),
    ("封建礼教", "职场PUA和服从文化"),
]

# ============================================================
# 鲁迅名言（用于原文评论类）
# ============================================================

FAMOUS_QUOTES = [
    "从来如此，便对么？",
    "不在沉默中爆发，就在沉默中灭亡",
    "真的猛士，敢于直面惨淡的人生",
    "哀其不幸，怒其不争",
    "愿中国青年都摆脱冷气",
    "人类的悲欢并不相通",
    "惟沉默是最高的轻蔑",
    "墨写的谎说，决掩不住血写的事实",
    "我向来是不惮以最坏的恶意来推测中国人的",
    "楼下一个男人病得要死，那间壁的一家唱着留声机",
]

QUOTE_COMMENTARY_TEMPLATES = [
    "先生，您说过「{quote}」。如果用这句话来评价{scenario}，您会怎么说？",
    "「{quote}」——这句话放在{scenario}的语境下，先生觉得还适用吗？",
    "鲁迅说过「{quote}」。有人拿这话来形容{scenario}，你同意吗？",
    "您的名言「{quote}」，放到2026年的{domain}领域来看，有什么新的含义？",
    "先生，「{quote}」这句话我一直记着。最近{scenario}，我突然又想起了这句话。",
    "如果用「{quote}」来总结现在{domain}的现状，先生觉得合适吗？",
]

# ============================================================
# 多轮对话开场白
# ============================================================

MULTI_TURN_STARTERS = [
    ("先生，我最近遇到一件事想跟您聊聊。", "什么事？说来听听。"),
    ("鲁迅先生，我有个困惑想请教您。", "嗯，你说。"),
    ("先生，我最近心情很差。", "怎么了？"),
    ("先生您好，我是一个大学生，想跟您谈谈我的迷茫。", "年轻人的迷茫，说来听听。"),
    ("先生，我今天看到一个新闻特别气愤。", "什么新闻？"),
    ("鲁迅先生，我跟朋友吵架了。", "因为什么？"),
    ("先生，我想辞职但又不敢。", "不敢的原因是什么？"),
    ("先生，您觉得现在的中国跟您那个时代比怎么样？", "说具体些。"),
]

MULTI_TURN_FOLLOWUPS = [
    "{scenario}。我不知道该怎么看待这件事。",
    "事情是这样的：{scenario}。您觉得我该怎么办？",
    "就是{scenario}。我身边的人都觉得这很正常，但我总觉得哪里不对。",
    "{scenario}。大家都说我想太多了，您觉得呢？",
    "具体来说就是{scenario}。我越想越觉得难受。",
    "{scenario}。我知道这不是什么大事，但它一直困扰着我。",
]


def generate_seeds(target_count: int = 5000) -> list[dict]:
    seeds = []

    all_concepts = []
    for theme, info in THEMES.items():
        for name, desc in info["concepts"]:
            all_concepts.append((theme, name, desc))

    domain_names = list(SCENARIOS.keys())

    # 1. 每个主题概念 x 每个场景领域 x ALL模板类型
    # Use a scenario index per concept to cycle through scenarios deterministically
    for ci, (theme, concept_name, concept_desc) in enumerate(all_concepts):
        for di, domain in enumerate(domain_names):
            scenario_list = SCENARIOS[domain]
            n_sc = len(scenario_list)
            # offset ensures different concepts pick different scenarios
            base = (ci * 7 + di * 3) % n_sc

            # 求助 - all templates
            for ti, tmpl in enumerate(TEMPLATES["求助"]):
                scenario = scenario_list[(base + ti) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(scenario=scenario),
                    "category": f"luxun_{theme}",
                    "theme": theme,
                    "domain": domain,
                })

            # 评论 - all templates
            for ti, tmpl in enumerate(TEMPLATES["评论"]):
                scenario = scenario_list[(base + ti + 3) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(scenario=scenario),
                    "category": f"luxun_{theme}",
                    "theme": theme,
                    "domain": domain,
                })

            # 深聊 - all templates
            for ti, tmpl in enumerate(TEMPLATES["深聊"]):
                scenario = scenario_list[(base + ti + 6) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(concept=concept_name, scenario=scenario),
                    "category": f"luxun_{theme}_deep",
                    "theme": theme,
                    "domain": domain,
                })

            # 犀利 - all templates
            for ti, tmpl in enumerate(TEMPLATES["犀利"]):
                scenario = scenario_list[(base + ti + 9) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(scenario=scenario),
                    "category": f"luxun_{theme}_sharp",
                    "theme": theme,
                    "domain": domain,
                })

            # 自省 - all templates, using concept + scenario
            for ti, tmpl in enumerate(TEMPLATES["自省"]):
                scenario = scenario_list[(base + ti + 12) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(concept=concept_name, scenario=scenario),
                    "category": f"luxun_{theme}_self",
                    "theme": theme,
                    "domain": domain,
                })

    # 2. 辩论类 - each saying x more concepts x all debate templates
    for saying in SAYINGS:
        for concept in random.sample(all_concepts, min(5, len(all_concepts))):
            for tmpl in TEMPLATES["辩论"]:
                seeds.append({
                    "prompt": tmpl.format(saying=saying, concept=concept[1]),
                    "category": "luxun_debate",
                    "theme": concept[0],
                    "domain": "辩论",
                })

    # 3. 对比类（古今对照）
    for old, new in OLD_NEW_PAIRS:
        for tmpl in TEMPLATES["对比"]:
            seeds.append({
                "prompt": tmpl.format(old_thing=old, new_thing=new),
                "category": "luxun_contrast",
                "theme": "古今",
                "domain": "对比",
            })

    # 4. 自省类 - all concepts x all self templates x domains
    for ci, concept in enumerate(all_concepts):
        for di, domain in enumerate(domain_names):
            scenario_list = SCENARIOS[domain]
            n_sc = len(scenario_list)
            tmpl = TEMPLATES["自省"][(ci + di) % len(TEMPLATES["自省"])]
            scenario = scenario_list[(ci * 3 + di * 7) % n_sc]
            seeds.append({
                "prompt": tmpl.format(concept=concept[1], scenario=scenario),
                "category": "luxun_self",
                "theme": concept[0],
                "domain": domain,
            })

    # 5. 闲聊
    for prompt in TEMPLATES["闲聊"]:
        seeds.append({
            "prompt": prompt,
            "category": "luxun_casual",
            "theme": "闲聊",
            "domain": "闲聊",
        })

    # 5b. 鲁迅原文评论 - take famous quotes and apply to modern scenarios
    for qi, quote in enumerate(FAMOUS_QUOTES):
        for di, domain in enumerate(domain_names):
            scenario_list = SCENARIOS[domain]
            n_sc = len(scenario_list)
            for ti, tmpl in enumerate(QUOTE_COMMENTARY_TEMPLATES):
                scenario = scenario_list[(qi * 3 + di * 5 + ti) % n_sc]
                seeds.append({
                    "prompt": tmpl.format(
                        quote=quote, scenario=scenario, domain=domain,
                    ),
                    "category": "luxun_quote_commentary",
                    "theme": "原文评论",
                    "domain": domain,
                })

    # 5c. 多轮对话开场白
    for si, (starter, reply) in enumerate(MULTI_TURN_STARTERS):
        for di, domain in enumerate(domain_names):
            scenario_list = SCENARIOS[domain]
            n_sc = len(scenario_list)
            for fi, followup_tmpl in enumerate(MULTI_TURN_FOLLOWUPS):
                scenario = scenario_list[(si * 5 + di * 3 + fi) % n_sc]
                prompt = f"{starter}\n\n（鲁迅：{reply}）\n\n{followup_tmpl.format(scenario=scenario)}"
                seeds.append({
                    "prompt": prompt,
                    "category": "luxun_multi_turn",
                    "theme": "多轮对话",
                    "domain": domain,
                })

    # 6. 补到目标数量 - use seen set to efficiently avoid duplicates
    seen_prompts = {s["prompt"] for s in seeds}
    all_scenarios_flat = []
    for d in domain_names:
        for sc in SCENARIOS[d]:
            all_scenarios_flat.append((d, sc))

    fill_templates = [
        ("求助", ["scenario"]),
        ("评论", ["scenario"]),
        ("犀利", ["scenario"]),
        ("深聊", ["concept", "scenario"]),
        ("辩论", ["saying", "concept"]),
        ("自省", ["concept", "scenario"]),
    ]

    max_fill_attempts = target_count * 20
    fill_attempt = 0
    while len(seeds) < target_count and fill_attempt < max_fill_attempts:
        fill_attempt += 1
        concept = random.choice(all_concepts)
        domain, scenario = random.choice(all_scenarios_flat)
        tmpl_type, _ = random.choice(fill_templates)
        tmpl = random.choice(TEMPLATES[tmpl_type])

        try:
            prompt = tmpl.format(
                scenario=scenario,
                concept=concept[1],
                saying=random.choice(SAYINGS),
                old_thing=random.choice(OLD_NEW_PAIRS)[0],
                new_thing=random.choice(OLD_NEW_PAIRS)[1],
            )
        except (KeyError, IndexError):
            continue

        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        seeds.append({
            "prompt": prompt,
            "category": f"luxun_{tmpl_type}",
            "theme": concept[0],
            "domain": domain,
        })

    # Dedup
    seen = set()
    unique = []
    for s in seeds:
        if s["prompt"] not in seen:
            seen.add(s["prompt"])
            unique.append(s)

    random.shuffle(unique)
    return unique[:target_count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/seeds/luxun_seeds.jsonl")
    parser.add_argument("--count", type=int, default=5000)
    args = parser.parse_args()

    seeds = generate_seeds(args.count)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    themes = Counter(s["theme"] for s in seeds)
    domains = Counter(s["domain"] for s in seeds)

    print(f"Generated {len(seeds)} seeds -> {args.output}")
    print(f"\nThemes: {themes.most_common(5)}")
    print(f"Domains: {domains.most_common(5)}")
