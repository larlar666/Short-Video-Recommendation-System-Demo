import sys, io
sys.path.insert(0, "../scripts")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tag_pipeline import _is_blacklisted, run_pipeline

cases = [
    # FYP Unicode 变体
    ("fypシ゚viral",           True),
    ("fypツ",                  True),
    ("fypage",                 True),
    ("fy",                     True),
    ("fypviral",               True),
    ("fyppppp",                True),
    ("foryoupage",             True),
    # 缩写残片
    ("n't",                    True),
    ("'re",                    True),
    # 平台词
    ("capcut",                 True),
    ("creatorsearchinsights",  True),
    ("replying",               True),
    ("viralvideo",             True),
    ("greenscreen",            True),
    # 短标签
    ("la",                     True),
    ("f",                      True),
    # 全非 ASCII（俄语/阿拉伯语 FYP 等价词）
    ("рек",                    True),   # 俄语"推荐"
    ("рекомендации",           True),   # 俄语"推荐"（长）
    ("اكسبلور",                True),   # 阿拉伯语"explore"
    # 混合内容：含 emoji 但有 ASCII 主体 → 保留
    ("usa🇺🇸",                 False),
    ("mexico🇲🇽",             False),
    # 正常标签
    ("dance",                  False),
    ("gym",                    False),
    ("nyc",                    False),
    ("mentalhealth",           False),
    ("storytime",              False),
    ("quinceañera",            False),  # 西班牙语，有 ASCII 主体，内容有意义
]

print("=== _is_blacklisted 测试 ===")
all_pass = True
for tag, expected in cases:
    result = _is_blacklisted(tag)
    status = "OK  " if result == expected else "FAIL"
    if status.strip() == "FAIL":
        all_pass = False
    print(f"  {status}  {repr(tag):<32}  期望={expected}  实际={result}")

print()
# 测试 run_pipeline 的 Step 5.5（全文直匹配）
print("=== run_pipeline Step 5.5 测试（无 POI，无 challenges）===")
row_no_poi = {
    "desc": "showing you the best gym workout routine for beginners",
    "challenges": "[]",
    "poi_category": "",
    "poi_tt_type_name_medium": "",
    "poi_tt_type_name_super": "",
}
result = run_pipeline(row_no_poi)
print(f"  fields: {result['fields']}")
print(f"  pref_tags: {result['preference_tags']}")
# 应包含 Physical Wellness & Fitness 或 Movement & Performance
assert any("Fitness" in f or "Performance" in f for f in result["fields"]), \
    "Step 5.5 未能从 desc 全文识别健身领域"

print()
print("全部通过！" if all_pass else "有失败项，请检查。")
