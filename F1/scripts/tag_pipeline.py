"""
tag_pipeline.py  —— 视频标签流水线

流程：
  1. 从 challenges（即 desc 中的 #话题）提取有效偏好标签
  2. 当 challenges 不足 10 个时，用 YAKE + TextRank 投票从 desc 正文补充
  3. 全部小写后执行前缀归约（保留最短串），并截断到 10 个
  4. 用 POI 字段映射到兴趣领域（最多 1 个）
  5. 用偏好标签的关键词投票补充兴趣领域，直到最多 3 个
  5.5 若领域仍不足，对 desc 全文做触发词直匹配（覆盖 pref_tags 漏检情况）
  6. （兜底）若领域仍不足 3 个，用 spaCy NER 识别 desc 中的专有名词类型补位
     PERSON→Pop Culture, ORG→Business, GPE/LOC→Travel, 等

用法示例（处理单行 CSV dict）：
    from tag_pipeline import run_pipeline
    result = run_pipeline(row)
    # result = {"preference_tags": [...], "fields": [...]}
"""

import json
import re
from collections import defaultdict
from typing import Optional

import yake
import summa  # TextRank，来自 summa 库
import spacy

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────

MAX_PREF_TAGS = 10   # 偏好标签上限
MAX_FIELDS = 3       # 兴趣领域上限

# 无意义 hashtag 黑名单（全小写，仅 ASCII 部分；含 Unicode 的变体在 _is_blacklisted 中
# 先剥去非 ASCII 字符后再比对，因此这里只需写 ASCII 形式）
HASHTAG_BLACKLIST: set[str] = {
    # ── FYP / foryou 系列 ────────────────────────
    "fyp", "fypsi", "fy", "fypage",
    "foryou", "foryoupage", "foryourpage",
    # ── 通用病毒/热门词 ──────────────────────────
    "viral", "viralvideo", "viraltiktok", "trending",
    "explore", "explorepage",
    # ── 平台/工具词 ──────────────────────────────
    "tiktok", "tiktokviral", "reels", "repost",
    "capcut",                   # 视频剪辑工具
    "creatorsearchinsights",    # TikTok SEO 工具标签
    "greenscreen",              # TikTok 特效，无内容含义
    "replying",                 # "Replying to @user" 的残留词
    # ── 其他泛化无意义词 ────────────────────────
    "parati", "xyzbca", "blowthisup", "blowup",
    "4u", "4you",
}

# 20 个兴趣领域
FIELDS = [
    "Movement & Performance",
    "Physical Wellness & Fitness",
    "Beauty & Personal Style",
    "Mental Health & Mindfulness",
    "Food & Culinary Arts",
    "Knowledge & Education",
    "Literature & Storytelling",
    "Finance & Wealth Building",
    "Entrepreneurship & Business",
    "Home & Domestic Life",
    "Digital Gaming & Technology",
    "Visual Arts & Design",
    "Travel & Exploration",
    "Relationships & Social Life",
    "Music & Sound Culture",
    "Comedy & Entertainment",
    "Nature & Sustainability",
    "Family & Parenting",
    "Pop Culture & Commentary",
    "DIY & Maker Culture",
]

# POI 类目 → 兴趣领域映射（模糊关键词匹配）
# key：POI 字符串中包含的关键词（小写）；value：对应的领域
POI_KEYWORD_MAP: list[tuple[str, str]] = [
    ("food",         "Food & Culinary Arts"),
    ("restaurant",   "Food & Culinary Arts"),
    ("cafe",         "Food & Culinary Arts"),
    ("bakery",       "Food & Culinary Arts"),
    ("bar",          "Food & Culinary Arts"),
    ("gym",          "Physical Wellness & Fitness"),
    ("fitness",      "Physical Wellness & Fitness"),
    ("sport",        "Movement & Performance"),
    ("dance",        "Movement & Performance"),
    ("salon",        "Beauty & Personal Style"),
    ("spa",          "Beauty & Personal Style"),
    ("beauty",       "Beauty & Personal Style"),
    ("hotel",        "Travel & Exploration"),
    ("resort",       "Travel & Exploration"),
    ("lodging",      "Travel & Exploration"),
    ("airport",      "Travel & Exploration"),
    ("museum",       "Knowledge & Education"),
    ("library",      "Knowledge & Education"),
    ("school",       "Knowledge & Education"),
    ("education",    "Knowledge & Education"),
    ("university",   "Knowledge & Education"),
    ("health",       "Mental Health & Mindfulness"),
    ("hospital",     "Mental Health & Mindfulness"),
    ("clinic",       "Mental Health & Mindfulness"),
    ("office",       "Entrepreneurship & Business"),
    ("industrial",   "Entrepreneurship & Business"),
    ("home",         "Home & Domestic Life"),
    ("garden",       "Home & Domestic Life"),
    ("game",         "Digital Gaming & Technology"),
    ("tech",         "Digital Gaming & Technology"),
    ("art",          "Visual Arts & Design"),
    ("gallery",      "Visual Arts & Design"),
    ("music",        "Music & Sound Culture"),
    ("theater",      "Comedy & Entertainment"),
    ("cinema",       "Comedy & Entertainment"),
    ("entertainment","Comedy & Entertainment"),
    ("park",         "Nature & Sustainability"),
    ("nature",       "Nature & Sustainability"),
]

# 偏好标签关键词 → 兴趣领域触发词表
# 每个领域配一组「若偏好标签中包含这些词则投票」的触发词
FIELD_TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "Movement & Performance":       ["dance", "dancing", "choreograph", "ballet", "perform", "cheer", "gymnastics"],
    "Physical Wellness & Fitness":  ["gym", "workout", "fitness", "yoga", "exercise", "run", "lifting", "bodybuilding"],
    "Beauty & Personal Style":      ["makeup", "skincare", "fashion", "ootd", "style", "beauty", "haircut", "nails"],
    "Mental Health & Mindfulness":  ["mentalhealth", "mindfulness", "anxiety", "selfcare", "therapy", "meditation"],
    "Food & Culinary Arts":         ["food", "cook", "recipe", "eat", "bake", "chef", "kitchen", "meal", "restaurant"],
    "Knowledge & Education":        ["learn", "education", "science", "fact", "history", "study", "school", "math"],
    "Literature & Storytelling":    ["book", "story", "poem", "reading", "writing", "novel", "author", "literature"],
    "Finance & Wealth Building":    ["money", "invest", "finance", "stock", "crypto", "wealth", "saving", "budget"],
    "Entrepreneurship & Business":  ["business", "startup", "entrepreneur", "hustle", "brand", "marketing", "career"],
    "Home & Domestic Life":         ["home", "decor", "cleaning", "interior", "house", "apartment", "organize"],
    "Digital Gaming & Technology":  ["gaming", "gamer", "minecraft", "fps", "tech", "coding", "programming", "ai"],
    "Visual Arts & Design":         ["art", "drawing", "painting", "design", "illustration", "sketch", "photography"],
    "Travel & Exploration":         ["travel", "trip", "vacation", "explore", "adventure", "flight", "wanderlust"],
    "Relationships & Social Life":  ["relationship", "couple", "dating", "friendship", "love", "marriage", "social"],
    "Music & Sound Culture":        ["music", "song", "sing", "rap", "hiphop", "guitar", "piano", "band", "concert"],
    "Comedy & Entertainment":       ["funny", "comedy", "humor", "prank", "joke", "lol", "meme", "skit"],
    "Nature & Sustainability":      ["nature", "wildlife", "plant", "eco", "green", "animal", "ocean", "forest"],
    "Family & Parenting":           ["mom", "dad", "baby", "toddler", "parenting", "kids", "family", "newborn"],
    "Pop Culture & Commentary":     ["celebrity", "movie", "drama", "kpop", "trending", "popculture", "review"],
    "DIY & Maker Culture":          ["diy", "craft", "handmade", "build", "maker", "woodwork", "sewing", "hack"],
}

# spaCy 命名实体类型 → 兴趣领域映射
# 当关键词触发词表未命中时，用 NER 识别出的实体类型做兜底推断
SPACY_ENTITY_TO_FIELD: dict[str, str] = {
    "PERSON":      "Pop Culture & Commentary",    # 人名（Justin Bieber, Jack Black）
    "ORG":         "Entrepreneurship & Business", # 品牌/组织（Lamborghini, Nike）
    "GPE":         "Travel & Exploration",        # 地缘政治实体（国家、城市）
    "LOC":         "Travel & Exploration",        # 地理位置（山脉、海洋）
    "PRODUCT":     "DIY & Maker Culture",         # 产品名（iPhone, PlayStation）
    "EVENT":       "Pop Culture & Commentary",    # 活动名（Super Bowl, World Cup）
    "WORK_OF_ART": "Pop Culture & Commentary",    # 作品名（电影、歌曲标题）
    "FAC":         "Travel & Exploration",        # 设施（建筑、机场）
}

# spaCy NLP 对象，模块级单例，首次调用时懒加载，避免每行都重新加载模型
_nlp: Optional[spacy.language.Language] = None
# YAKE 提取器单例，同理避免每行重复初始化
_yake_extractor: Optional[yake.KeywordExtractor] = None


def _get_nlp() -> spacy.language.Language:
    """懒加载 spaCy 模型，整个进程只加载一次。"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def _get_yake() -> yake.KeywordExtractor:
    """懒加载 YAKE 提取器，整个进程只初始化一次。"""
    global _yake_extractor
    if _yake_extractor is None:
        _yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=1,
            dedupLim=0.7,
            top=MAX_PREF_TAGS * 3,  # 固定候选池大小（30），每次调用后按需截断
        )
    return _yake_extractor


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _normalize_tag(tag: str) -> str:
    """规范化单个 tag：去首尾空白、去开头 #、转小写。"""
    return tag.strip().lstrip("#").lower()


def _is_blacklisted(tag: str) -> bool:
    """
    判断一个已规范化的 tag 是否无意义，有以下五条规则：

    1. 长度 < 3：单字母/双字母标签几乎均无内容语义（'fy', 'la', 'f' 等）。
    2. 缩略词残片：如 "n't"、"'re"、"'ve"、"'ll"（英文缩略词被 YAKE 切词后的残余）。
    3. 全为非 ASCII 字符：西里尔/阿拉伯等语言的"推荐"系标签（рек、اكسبلور 等），
       剥去非 ASCII 后字符串为空，说明该 tag 整体不含英文内容，过滤掉。
       注：含混合内容的标签（usa🇺🇸 → ascii_tag="usa"）不受影响，会保留。
    4. 字面黑名单命中（含 Unicode 清洗后比对）：
       先用正则剥去非 ASCII 字符，再检查 HASHTAG_BLACKLIST。
       这样 "fypシ゚viral"→"fypviral"、"fypツ"→"fyp" 等 Unicode 变体都能被捕捉。
    5. FYP / foryou 重复字母变体：正则覆盖 "fyppppp"、"fyyyyp"、"forrrryou" 等，
       以及剥去 Unicode 后以 "fyp" 开头的一切形式（"fypviral"、"fyppage" 等）。
    """
    # 规则 1：过短
    if len(tag) < 3:
        return True

    # 规则 2：缩略词残片（以撇号开头，或仅一个字母 + 撇号 + 字母的形式，如 n't）
    if re.fullmatch(r"[a-z]?'[a-z]+", tag):
        return True

    # 规则 3：剥去非 ASCII 后为空 → 全非 ASCII 标签（俄语/阿拉伯语 FYP 同类词等）
    ascii_tag = re.sub(r"[^\x00-\x7F]", "", tag)  # 去掉日语假名、emoji、西里尔字母等
    if not ascii_tag:
        return True

    # 规则 4：字面黑名单（使用 ascii_tag 使 Unicode 变体也能命中）
    if tag in HASHTAG_BLACKLIST or ascii_tag in HASHTAG_BLACKLIST:
        return True

    # 规则 5：FYP / foryou 变体（含字母重复拉伸 和 fyp 打头的组合词）
    if re.fullmatch(r"f+y+p+", ascii_tag):         # fyppppp、fyyyyp 等
        return True
    if re.fullmatch(r"f+o+r+y+o+u+.*", ascii_tag): # foryou、forrrryou、foryoupage 等
        return True
    if ascii_tag.startswith("fyp"):                 # fypviral、fyppage、fypシ゚viral 等
        return True

    # 规则 6：含标点符号（仅允许字母、数字、空格）
    if not re.fullmatch(r"[a-z0-9 ]+", tag):
        return True

    return False


def _parse_challenges(raw_challenges) -> list[str]:
    """
    将 CSV 里存储的 challenges 字段（JSON 字符串或列表）解析为 tag 字符串列表。
    TikTok-10M 中 challenges 每个元素是一个 dict，通常含 "title" 键。
    """
    if not raw_challenges:
        return []

    # CSV 导出时已 json.dumps，读回来是字符串
    if isinstance(raw_challenges, str):
        try:
            data = json.loads(raw_challenges)
        except json.JSONDecodeError:
            return []
    else:
        data = raw_challenges  # 直接是列表（流式读取时）

    tags = []
    for item in data:
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or ""
        elif isinstance(item, str):
            title = item
        else:
            continue
        tags.append(title)

    return tags


def _extract_tags_from_desc_with_voting(
    clean_text: str, n_needed: int
) -> list[str]:
    """
    用 YAKE + TextRank 投票从正文文本中提取关键词。

    投票规则：
      - YAKE 和 TextRank 各自给出 top-30 候选词（固定候选池）
      - 同时出现在两个结果中的词得 2 票，否则得 1 票
      - 优先保留 2 票词；2 票不足时，按 YAKE 得分（更稳定）补位
    """
    if not clean_text or len(clean_text.split()) < 3:
        return []

    candidate_pool = n_needed * 3  # 候选池大小

    # ── YAKE ─────────────────────────────────
    # yake 返回 [(词, 分值), ...]，分值越小越重要
    yake_results = _get_yake().extract_keywords(clean_text)
    # 转成 {词: 重要度}，重要度 = 1/(1+分值) 使其变为「越大越好」
    yake_scores: dict[str, float] = {
        _normalize_tag(w): 1.0 / (1.0 + s)
        for w, s in yake_results
    }

    # ── TextRank（summa 库）───────────────────
    try:
        tr_output = summa.keywords(clean_text, words=candidate_pool, split=True)
    except Exception:
        tr_output = []
    tr_words: set[str] = {_normalize_tag(w) for w in tr_output}

    # ── 投票融合 ─────────────────────────────
    # 2 票：YAKE 和 TextRank 都选中的词
    two_vote = [w for w in yake_scores if w in tr_words]
    # 1 票：只有 YAKE 选中（TextRank 结果不稳定，不单独用 TextRank 补）
    one_vote = [w for w in yake_scores if w not in tr_words]

    # 2 票词按 YAKE 重要度降序排列（它俩都认可，YAKE 分更细粒度）
    two_vote.sort(key=lambda w: yake_scores[w], reverse=True)
    one_vote.sort(key=lambda w: yake_scores[w], reverse=True)

    # 合并：先放 2 票，再用 1 票补位
    combined = two_vote + one_vote
    return combined[:n_needed]


def _deduplicate_by_substring(tags: list[str]) -> list[str]:
    """
    前缀归约：若 tag B 以 tag A 为前缀（B 以 A 开头且 B ≠ A），删除 B，保留更短的 A。
    只允许前缀关系，避免短词误伤语义无关的长词。

    例：["cat", "cats", "catsoftiktok"] → ["cat"]
        ["cooking", "cookingvideo"]     → ["cooking"]
        ["art", "party"]                → ["art", "party"]  （不误伤）
        ["mom", "momentum"]             → ["mom", "momentum"]（不误伤）
    """
    to_remove: set[str] = set()
    for i, tag_a in enumerate(tags):
        for j, tag_b in enumerate(tags):
            if i == j:
                continue
            # 只在 tag_b 以 tag_a 为前缀时才认为有语义延伸关系
            if tag_b.startswith(tag_a) and tag_b != tag_a:
                to_remove.add(tag_b)  # 删长的，保留短的 tag_a

    return [t for t in tags if t not in to_remove]


# ──────────────────────────────────────────────
# 核心流水线函数
# ──────────────────────────────────────────────

def extract_preference_tags(row: dict) -> list[str]:
    """
    Step 1+2：生成偏好标签列表（≤ MAX_PREF_TAGS 个）。

    优先从 challenges 中提取有效 hashtag；不足时用 YAKE+TextRank 从 desc 正文补充。
    最后对全部候选做前缀归约，截断到上限。
    """
    # ── Step 1：从 challenges 提取有效 hashtag ──
    raw_challenges = row.get("challenges")
    raw_tags = _parse_challenges(raw_challenges)

    valid_challenge_tags: list[str] = []
    for raw in raw_tags:
        tag = _normalize_tag(raw)
        if tag and not _is_blacklisted(tag):
            valid_challenge_tags.append(tag)

    # challenges 最多保留前 MAX_PREF_TAGS 个（防止极端数量）
    valid_challenge_tags = valid_challenge_tags[:MAX_PREF_TAGS]

    # ── Step 2：从 desc 正文提取补充关键词（仅当不足 10 个时）──
    desc: str = row.get("desc") or ""

    desc_no_hashtags = re.sub(r"#\S+", "", desc)            # 删掉所有 #话题
    desc_clean = re.sub(r"@\S+", "", desc_no_hashtags).strip()  # 删掉所有 @提及

    derived_tags: list[str] = []
    n_needed = MAX_PREF_TAGS - len(valid_challenge_tags)
    if n_needed > 0 and desc_clean:
        raw_derived = _extract_tags_from_desc_with_voting(desc_clean, n_needed)
        for tag in raw_derived:
            tag = _normalize_tag(tag)
            if tag and not _is_blacklisted(tag) and tag not in valid_challenge_tags:
                derived_tags.append(tag)

    # ── Step 3：合并 + 前缀归约 + 截断 ──
    all_candidates = valid_challenge_tags + derived_tags
    deduplicated = _deduplicate_by_substring(all_candidates)
    return deduplicated[:MAX_PREF_TAGS]


def map_poi_to_field(row: dict) -> str | None:
    """
    Step 4：将 POI 字段映射到至多 1 个兴趣领域。

    优先查 poi_tt_type_name_medium，降级查 poi_category，再降级查 poi_tt_type_name_super。
    使用模糊关键词匹配（只要 POI 字符串包含关键词即命中），取第一个命中的领域。
    若无法映射则返回 None。
    """
    poi_values = [
        row.get("poi_tt_type_name_medium") or "",
        row.get("poi_category") or "",
        row.get("poi_tt_type_name_super") or "",
    ]

    for poi_raw in poi_values:
        poi = poi_raw.lower().strip()
        if not poi:
            continue
        for keyword, field in POI_KEYWORD_MAP:
            if keyword in poi:
                return field  # 取第一个命中，立即返回

    return None  # 无法映射


def map_tags_to_fields(
    pref_tags: list[str], existing_fields: list[str]
) -> list[str]:
    """
    Step 5：用偏好标签的关键词投票，补充兴趣领域，直到总数达到 MAX_FIELDS。

    投票逻辑：遍历每个偏好标签，检查它是否包含某个领域的触发关键词；
    若命中则给该领域加 1 票。最后按票数降序，依次补入未出现的领域。
    """
    needed = MAX_FIELDS - len(existing_fields)
    if needed <= 0:
        return []

    vote_counts: dict[str, int] = defaultdict(int)
    for tag in pref_tags:
        for field, keywords in FIELD_TRIGGER_KEYWORDS.items():
            if field in existing_fields:
                continue  # 该领域已有，无需重复投票
            for kw in keywords:
                if kw in tag:  # tag 中包含触发词
                    vote_counts[field] += 1
                    break  # 一个 tag 对同一领域只投 1 票

    ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    new_fields = [f for f, v in ranked if v > 0]
    return new_fields[:needed]


def _desc_text_to_fields(clean_desc: str, existing_fields: list[str]) -> list[str]:
    """
    Step 5.5（全文直匹配）：直接在清洗后的 desc 文本中搜索触发词，补充兴趣领域。

    与 Step 5 的区别：Step 5 只扫描 pref_tags 列表（已去噪后的 10 个标签），
    而这一步扫描整段 desc 原文，覆盖"desc 有内容但 YAKE/TextRank 提取的词恰好
    不在触发词表"的情况，是 NER 之前的轻量级补位手段。

    参数
    ----
    clean_desc : str    已去掉 # 和 @ 的 desc 文本（全小写）
    existing_fields     已有的兴趣领域列表
    """
    needed = MAX_FIELDS - len(existing_fields)
    if needed <= 0 or not clean_desc:
        return []

    vote_counts: dict[str, int] = defaultdict(int)
    for field, keywords in FIELD_TRIGGER_KEYWORDS.items():
        if field in existing_fields:
            continue
        for kw in keywords:
            if kw in clean_desc:          # 在 desc 全文中匹配触发词
                vote_counts[field] += 1
                break                     # 一个领域只投 1 票

    ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    new_fields = [f for f, v in ranked if v > 0]
    return new_fields[:needed]


def _ner_to_fields(desc: str, existing_fields: list[str]) -> list[str]:
    """
    Step 6（兜底）：对 desc 原文运行 spaCy NER，将识别出的实体类型映射到兴趣领域。

    只在 Step 5 之后仍不足 MAX_FIELDS 个领域时调用，作为最后的补位手段。
    NER 在完整 desc（含 #话题、@提及）上运行，因为专有名词通常在这些部分出现。

    映射规则：
      PERSON       → Pop Culture & Commentary   （人名：网红、明星、运动员）
      ORG          → Entrepreneurship & Business （品牌/机构：Lamborghini、Nike）
      GPE / LOC    → Travel & Exploration        （地名：城市、国家）
      PRODUCT      → DIY & Maker Culture         （产品名：iPhone、PlayStation）
      EVENT        → Pop Culture & Commentary    （赛事/活动：Super Bowl）
      WORK_OF_ART  → Pop Culture & Commentary    （作品名：电影、歌名）
      FAC          → Travel & Exploration        （建筑/设施：Eiffel Tower）
    """
    needed = MAX_FIELDS - len(existing_fields)
    if needed <= 0 or not desc:
        return []

    nlp = _get_nlp()
    doc = nlp(desc)

    vote_counts: dict[str, int] = defaultdict(int)
    for ent in doc.ents:
        field = SPACY_ENTITY_TO_FIELD.get(ent.label_)
        if field and field not in existing_fields:
            vote_counts[field] += 1

    ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    new_fields = [f for f, _ in ranked if f not in existing_fields]
    return new_fields[:needed]


def run_pipeline(row: dict) -> dict:
    """
    统一入口：对单行数据运行完整标签流水线。

    参数
    ----
    row : dict
        一条视频记录，至少包含 "desc"、"challenges"、"poi_category" 等字段。

    返回
    ----
    dict，包含：
        preference_tags : list[str]   偏好标签（≤10 个）
        fields          : list[str]   兴趣领域（≤3 个）
    """
    # Step 1+2+3：偏好标签
    pref_tags = extract_preference_tags(row)

    # Step 4：POI → 兴趣领域
    poi_field = map_poi_to_field(row)
    fields: list[str] = [poi_field] if poi_field else []

    # Step 5：偏好标签关键词投票补充领域
    extra_fields = map_tags_to_fields(pref_tags, fields)
    fields = fields + extra_fields

    # Step 5.5：若领域仍不足，对 desc 全文做触发词直匹配
    # 这一步覆盖"desc 有内容但 pref_tags 过少或已被过滤干净"的情况
    if len(fields) < MAX_FIELDS:
        desc = row.get("desc") or ""
        desc_clean = re.sub(r"[#@]\S+", "", desc).lower().strip()
        text_fields = _desc_text_to_fields(desc_clean, fields)
        fields = fields + text_fields

    # Step 6（兜底）：若领域仍不足 MAX_FIELDS，用 NER 识别专有名词补位
    if len(fields) < MAX_FIELDS:
        desc = row.get("desc") or ""
        ner_fields = _ner_to_fields(desc, fields)
        fields = fields + ner_fields

    return {
        "preference_tags": pref_tags,
        "fields": fields,
    }
