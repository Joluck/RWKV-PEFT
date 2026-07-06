#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-Tuning Memory Probe

- 只训练 RWKV 的 `time_state` / `ts_state`
- 删除 math 判题、外部 judge 与旧强化学习训练逻辑
- 删除 prompt 中的 think 标签
- 内置一组 2026 年虚构知识问答，用于测试 state-tuning 的记忆能力
- 支持从 1 个知识开始，按课程式逐步增加知识条目数量
"""

import os
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(REPO_DIR, "reference")
SOTA_REFERENCE_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "inference", "sota", "reference"))
RAPID_SAMPLING_DIR = os.path.join(REPO_DIR, "Rapid-Sampling-main")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, REFERENCE_DIR)
sys.path.insert(0, SOTA_REFERENCE_DIR)

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_current_cuda_device(device: str):
    if not torch.cuda.is_available():
        return
    if not isinstance(device, str) or not device.startswith("cuda"):
        return
    if ":" in device:
        idx = int(device.split(":", 1)[1])
    else:
        idx = 0
    if torch.cuda.current_device() != idx:
        torch.cuda.set_device(idx)


def _normalize_record(rec: Any) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        return {"problem": str(rec), "solution": ""}
    out = dict(rec)
    if not out.get("problem"):
        for key in ("question", "prompt", "input"):
            if out.get(key):
                out["problem"] = out[key]
                break
    if not out.get("solution"):
        for key in ("answer", "output", "target"):
            if out.get(key):
                out["solution"] = out[key]
                break
    out.setdefault("problem", "")
    out.setdefault("solution", "")
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(_normalize_record(json.loads(line)))
    return data


def load_data(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [_normalize_record(x) for x in raw]
    raise ValueError(f"Unsupported data format: {path}")


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("</s>", "")
    text = text.replace("<s>", "")
    return text.strip()


def normalize_answer(text: str) -> str:
    text = _clean_text(text)
    text = text.replace("Assistant:", "").strip()
    text = text.replace("：", ":")
    text = re.sub(r"\s+", "", text)
    text = text.strip("。.,，!！?？;；:：\"'`()[]{}")
    return text.lower()


def extract_answer_text(text: str) -> str:
    text = _clean_text(text)
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0]
    if "\n\nUser:" in text:
        text = text.split("\n\nUser:", 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        candidate = line.strip()
        if candidate in {"Assistant:", "Assistant"}:
            continue
        if candidate.startswith("Assistant:"):
            candidate = candidate[len("Assistant:"):].strip()
        if not candidate:
            continue
        return candidate.strip()
    return ""


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


# =========================================================
# Synthetic 2026 knowledge bank
# =========================================================

@dataclass
class MemoryFact:
    fact_id: str
    answer: str
    train_questions: List[str]
    eval_questions: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


def build_synthetic_2026_facts() -> List[MemoryFact]:
    return [
        MemoryFact(
            fact_id="fact_01",
            answer="雾栗",
            train_questions=[
                "2026年云桥市记忆节的吉祥物叫什么？",
                "请直接回答：2026年云桥市记忆节的吉祥物名字是？",
                "如果别人问你云桥市 2026 记忆节 mascot 是什么，你该回答什么？",
                "云桥市在2026年举办的记忆节，官方吉祥物名称是什么？",
                "只回答名称：2026 云桥记忆节吉祥物。",
                "2026年的云桥记忆节对应哪个吉祥物？",
            ],
            eval_questions=[
                "2026 云桥市记忆节的官方 mascot 叫什么名字？",
                "2026年云桥记忆节用的吉祥物名称是啥？",
                "说出 2026 年云桥市记忆节的吉祥物。",
                "请问 2026 云桥记忆节吉祥物叫什么？",
            ],
            meta={"topic": "festival_mascot"},
        ),
        MemoryFact(
            fact_id="fact_02",
            answer="73B",
            train_questions=[
                "2026年澄港夜航节的备用广播频道编号是什么？",
                "请直接回答：2026 澄港夜航节备用频道号。",
                "如果有人询问2026年澄港夜航节的备用频道，你该回答哪个编号？",
                "澄港在2026年夜航节启用的备用广播频道是多少？",
                "只说编号：2026 澄港夜航节备用频道。",
                "2026年的澄港夜航节，备用广播频道编号是？",
            ],
            eval_questions=[
                "2026 澄港夜航节的备用频道是哪个号码？",
                "澄港夜航节在 2026 年使用的备用广播编号是什么？",
                "告诉我 2026 年澄港夜航节备用频道号。",
                "2026年澄港夜航节备用频道编号叫什么？",
            ],
            meta={"topic": "channel_code"},
        ),
        MemoryFact(
            fact_id="fact_03",
            answer="海盐柚茶",
            train_questions=[
                "2026年折星列车头等舱的欢迎饮品是什么？",
                "请直接回答：2026 折星列车头等舱欢迎饮品。",
                "2026年的折星列车，一等舱上车后先提供什么欢迎饮品？",
                "如果提到 2026 折星列车头等舱欢迎饮品，你应该回答什么？",
                "只回答饮品名：2026 折星列车头等舱。",
                "2026年折星列车头等舱默认欢迎饮品叫什么？",
            ],
            eval_questions=[
                "2026 折星列车头等舱的 welcome drink 是什么？",
                "2026年折星列车一等舱欢迎饮品名称是？",
                "说出 2026 年折星列车头等舱的迎宾饮品。",
                "2026 折星列车头等舱先送上的饮品是什么？",
            ],
            meta={"topic": "drink"},
        ),
        MemoryFact(
            fact_id="fact_04",
            answer="青陀螺",
            train_questions=[
                "2026年北屿微型机器人赛的冠军队名是什么？",
                "请直接回答：2026 北屿微型机器人赛冠军队名。",
                "北屿在2026年的微型机器人赛，冠军队伍叫什么？",
                "如果别人问 2026 北屿微型机器人赛是谁夺冠，你该回答哪支队名？",
                "只说队名：2026 北屿微型机器人赛冠军。",
                "2026年的北屿微型机器人赛冠军名称是？",
            ],
            eval_questions=[
                "2026 北屿微型机器人赛冠军队叫啥？",
                "告诉我 2026 年北屿微型机器人赛冠军队名。",
                "说出 2026 北屿微型机器人赛夺冠队伍名称。",
                "2026年北屿机器人小赛的冠军队是什么名字？",
            ],
            meta={"topic": "team_name"},
        ),
        MemoryFact(
            fact_id="fact_05",
            answer="先听风，再开门",
            train_questions=[
                "2026年岚汀数据馆的入馆口令是什么？",
                "请直接回答：2026 岚汀数据馆入馆口令。",
                "岚汀数据馆在2026年使用的入馆口令是哪一句？",
                "如果别人问你 2026 岚汀数据馆进门口令，你应该回答什么？",
                "只回答口令：2026 岚汀数据馆。",
                "2026年的岚汀数据馆，入馆口令是什么句子？",
            ],
            eval_questions=[
                "2026 岚汀数据馆的门禁口令是什么？",
                "说出 2026 年岚汀数据馆入馆口令。",
                "2026年岚汀数据馆的进馆口令是哪句话？",
                "请问 2026 岚汀数据馆口令是什么？",
            ],
            meta={"topic": "passphrase"},
        ),
        MemoryFact(
            fact_id="fact_06",
            answer="雾青",
            train_questions=[
                "2026年白砂天线节的主色叫什么？",
                "请直接回答：2026 白砂天线节主色名。",
                "白砂天线节在2026年的主视觉颜色是什么？",
                "如果提到 2026 白砂天线节的主色，你该回答哪个名字？",
                "只回答颜色名：2026 白砂天线节。",
                "2026年的白砂天线节采用什么主色？",
            ],
            eval_questions=[
                "2026 白砂天线节的主视觉色叫什么？",
                "说出 2026 年白砂天线节的主色。",
                "2026年白砂天线节主色名称是？",
                "白砂天线节在 2026 用的主色是什么？",
            ],
            meta={"topic": "color"},
        ),
        MemoryFact(
            fact_id="fact_07",
            answer="05:17",
            train_questions=[
                "2026年回声码头的凌晨公告时刻是几点？",
                "请直接回答：2026 回声码头凌晨公告时刻。",
                "回声码头在2026年的固定凌晨公告时间是多少？",
                "如果别人问 2026 回声码头凌晨几点发公告，你该回答什么时间？",
                "只回答时间：2026 回声码头凌晨公告。",
                "2026年的回声码头，凌晨公告时刻是哪一个？",
            ],
            eval_questions=[
                "2026 回声码头凌晨公告时间是？",
                "告诉我 2026 年回声码头固定凌晨公告时刻。",
                "2026年回声码头的凌晨公告在几点发布？",
                "说出 2026 回声码头凌晨公告的具体时刻。",
            ],
            meta={"topic": "time"},
        ),
        MemoryFact(
            fact_id="fact_08",
            answer="K-Blue9",
            train_questions=[
                "2026年星槐研究站的内部代号是什么？",
                "请直接回答：2026 星槐研究站内部代号。",
                "星槐研究站在2026年使用的内部代号叫什么？",
                "如果别人问 2026 星槐研究站代号，你该回答什么？",
                "只回答代号：2026 星槐研究站。",
                "2026年的星槐研究站，内部代号是？",
            ],
            eval_questions=[
                "2026 星槐研究站对应的内部编号是什么？",
                "说出 2026 年星槐研究站的代号。",
                "2026年星槐研究站内部代号叫什么名字？",
                "请问 2026 星槐研究站的内部 code 是什么？",
            ],
            meta={"topic": "station_code"},
        ),
        MemoryFact(
            fact_id="fact_09",
            answer="折光七号",
            train_questions=[
                "2026年远汐港试运行的小渡船名字是什么？",
                "请直接回答：2026 远汐港试运行渡船名。",
                "远汐港在2026年测试的小型渡船叫什么？",
                "如果别人问 2026 远汐港试运行渡船名称，你该回答什么？",
                "只回答船名：2026 远汐港试运行渡船。",
                "2026年的远汐港，测试渡船名称是？",
            ],
            eval_questions=[
                "2026 远汐港试运行渡船叫什么？",
                "说出 2026 年远汐港试运行的小渡船名字。",
                "2026年远汐港测试渡船的名称是什么？",
                "请问 2026 远汐港那艘试运行渡船叫什么名字？",
            ],
            meta={"topic": "ship_name"},
        ),
        MemoryFact(
            fact_id="fact_10",
            answer="Luma-3",
            train_questions=[
                "2026年霁河档案库使用的巡检无人机型号是什么？",
                "请直接回答：2026 霁河档案库巡检无人机型号。",
                "霁河档案库在2026年配备的巡检无人机叫什么型号？",
                "如果提到 2026 霁河档案库巡检无人机，你该回答哪个型号？",
                "只回答型号：2026 霁河档案库巡检机。",
                "2026年的霁河档案库，巡检无人机型号是？",
            ],
            eval_questions=[
                "2026 霁河档案库的巡检无人机型号是什么？",
                "说出 2026 年霁河档案库使用的无人机型号。",
                "2026年霁河档案库巡检机叫什么？",
                "请问 2026 霁河档案库巡检无人机的型号是啥？",
            ],
            meta={"topic": "drone_model"},
        ),
        MemoryFact(
            fact_id="fact_11",
            answer="银槭路17号",
            train_questions=[
                "2026年浮岚电台临时演播室设在哪个地址？",
                "请直接回答：2026 浮岚电台临时演播室地址。",
                "浮岚电台在2026年的临时演播室地址是什么？",
                "如果别人问 2026 浮岚电台临时演播室在哪，你该回答哪个地址？",
                "只回答地址：2026 浮岚电台临时演播室。",
                "2026年的浮岚电台，临时演播室设在什么地方？",
            ],
            eval_questions=[
                "2026 浮岚电台临时演播室的地址是什么？",
                "说出 2026 年浮岚电台临时演播室地址。",
                "2026年浮岚电台临时演播室在哪条路几号？",
                "请问 2026 浮岚电台临时演播室设在哪？",
            ],
            meta={"topic": "address"},
        ),
        MemoryFact(
            fact_id="fact_12",
            answer="回形砂",
            train_questions=[
                "2026年临穹美术周主展厅地面的纹样代号是什么？",
                "请直接回答：2026 临穹美术周主展厅地面纹样代号。",
                "临穹美术周在2026年主展厅地板纹样叫什么？",
                "如果别人问 2026 临穹美术周主展厅纹样代号，你该回答什么？",
                "只回答代号：2026 临穹美术周主展厅纹样。",
                "2026年的临穹美术周，主展厅地面纹样代号是？",
            ],
            eval_questions=[
                "2026 临穹美术周主展厅地面纹样叫什么？",
                "说出 2026 年临穹美术周主展厅地面纹样代号。",
                "2026年临穹美术周主展厅纹样名称是？",
                "请问 2026 临穹美术周主展厅那套地面纹样叫什么？",
            ],
            meta={"topic": "pattern_code"},
        ),
        MemoryFact(
            fact_id="fact_13",
            answer="9月17日",
            train_questions=[
                "2026年镜潮书展的开幕日是哪一天？",
                "请直接回答：2026 镜潮书展开幕日期。",
                "镜潮书展在2026年是哪天开幕的？",
                "如果别人问 2026 镜潮书展开幕日，你该回答哪一天？",
                "只回答日期：2026 镜潮书展。",
                "2026年的镜潮书展开幕时间是几月几日？",
            ],
            eval_questions=[
                "2026 镜潮书展的开幕日是什么时候？",
                "说出 2026 年镜潮书展开幕日期。",
                "2026年镜潮书展在哪一天开幕？",
                "请问 2026 镜潮书展开幕是几月几号？",
            ],
            meta={"topic": "date"},
        ),
        MemoryFact(
            fact_id="fact_14",
            answer="白曜塔",
            train_questions=[
                "2026年浅湾天文夜的主观测塔叫什么？",
                "请直接回答：2026 浅湾天文夜主观测塔名称。",
                "浅湾天文夜在2026年的主观测塔名字是什么？",
                "如果别人问 2026 浅湾天文夜主观测塔，你该回答什么？",
                "只回答塔名：2026 浅湾天文夜主观测塔。",
                "2026年的浅湾天文夜，主观测塔叫啥？",
            ],
            eval_questions=[
                "2026 浅湾天文夜主观测塔叫什么名字？",
                "说出 2026 年浅湾天文夜的主观测塔。",
                "2026年浅湾天文夜主塔名称是什么？",
                "请问 2026 浅湾天文夜使用的主观测塔叫什么？",
            ],
            meta={"topic": "tower_name"},
        ),
        MemoryFact(
            fact_id="fact_15",
            answer="珀铃",
            train_questions=[
                "2026年听澜酒店大厅香氛的名字是什么？",
                "请直接回答：2026 听澜酒店大厅香氛名称。",
                "听澜酒店在2026年大厅使用的香氛叫什么？",
                "如果有人问 2026 听澜酒店大厅香氛，你该回答什么名字？",
                "只回答香氛名：2026 听澜酒店大厅。",
                "2026年的听澜酒店，大厅香氛名称是？",
            ],
            eval_questions=[
                "2026 听澜酒店大厅香氛叫什么？",
                "说出 2026 年听澜酒店大厅香氛名称。",
                "2026年听澜酒店大厅用的香氛是什么？",
                "请问 2026 听澜酒店大堂香氛名字是？",
            ],
            meta={"topic": "scent"},
        ),
        MemoryFact(
            fact_id="fact_16",
            answer="R-218",
            train_questions=[
                "2026年岑光实验园温室二区的登记编号是什么？",
                "请直接回答：2026 岑光实验园温室二区登记编号。",
                "岑光实验园在2026年温室二区的编号叫什么？",
                "如果别人问 2026 岑光实验园温室二区编号，你该回答什么？",
                "只回答编号：2026 岑光实验园温室二区。",
                "2026年的岑光实验园，温室二区登记编号是？",
            ],
            eval_questions=[
                "2026 岑光实验园温室二区的登记编号是什么？",
                "说出 2026 年岑光实验园温室二区编号。",
                "2026年岑光实验园温室二区叫什么编号？",
                "请问 2026 岑光实验园温室二区的 code 是什么？",
            ],
            meta={"topic": "greenhouse_code"},
        ),
    ]


def build_memory_dataset(facts: List[MemoryFact], split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fact in facts:
        questions = fact.train_questions if split == "train" else fact.eval_questions
        for idx, q in enumerate(questions):
            rows.append({
                "problem": q,
                "solution": fact.answer,
                "fact_id": fact.fact_id,
                "variant_id": idx,
                "split": split,
                "meta": fact.meta,
            })
    return rows


def parse_fact_schedule(spec: str, max_facts: int) -> List[int]:
    values: List[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        num = int(part)
        if not (1 <= num <= max_facts):
            raise ValueError(f"fact schedule item out of range: {num}")
        if not values or values[-1] != num:
            values.append(num)
    if not values:
        values = [1]
    return values


def build_curriculum(schedule: List[int], all_facts: List[MemoryFact]) -> List[Dict[str, Any]]:
    stages = []
    for stage_idx, fact_count in enumerate(schedule, start=1):
        facts = all_facts[:fact_count]
        stages.append({
            "stage_idx": stage_idx,
            "fact_count": fact_count,
            "facts": facts,
            "train": build_memory_dataset(facts, split="train"),
            "eval": build_memory_dataset(facts, split="eval"),
        })
    return stages


def dump_curriculum_data(out_dir: str, stages: List[Dict[str, Any]]):
    data_dir = os.path.join(out_dir, "datasets")
    os.makedirs(data_dir, exist_ok=True)
    manifest = []
    for stage in stages:
        fact_count = int(stage["fact_count"])
        train_path = os.path.join(data_dir, f"synthetic_2026_k{fact_count}_train.jsonl")
        eval_path = os.path.join(data_dir, f"synthetic_2026_k{fact_count}_eval.jsonl")
        write_jsonl(train_path, stage["train"])
        write_jsonl(eval_path, stage["eval"])
        manifest.append({
            "stage_idx": stage["stage_idx"],
            "fact_count": fact_count,
            "train_path": train_path,
            "eval_path": eval_path,
            "fact_ids": [fact.fact_id for fact in stage["facts"]],
        })
    with open(os.path.join(data_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# =========================================================
# Prompt
# =========================================================

def build_prompt(problem: str, prompt_format: str = "chat") -> str:
    p = (problem or "").strip()
    fmt = (prompt_format or "chat").strip().lower()
    if fmt == "completion":
        return f"Question: {p} Answer:"
    return (
        "你是一个问答助手。请直接回答，不要解释，不要重复问题。\n"
        f"User: {p}\n"
        "Assistant:"
    )


def build_item_prompt(item: Dict[str, Any], prompt_format: str = "chat") -> str:
    base_prompt = build_prompt(item.get("problem", ""), prompt_format=prompt_format)
    prefix = str(item.get("context_prefix", "") or item.get("prefix_text", "") or "").strip()
    if not prefix:
        return base_prompt
    if prompt_format == "completion":
        return prefix + "\n" + base_prompt
    return prefix + "\n\n" + base_prompt


def extract_completion_answer(text: str) -> str:
    text = _clean_text(text)
    text = re.split(r"\n+\s*Q(?:uestion)?:", text, maxsplit=1)[0]
    text = re.split(r"\n+\s*A(?:nswer)?:", text, maxsplit=1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    for prefix in ["Answer:", "A:"]:
        if first.startswith(prefix):
            first = first[len(prefix):].strip()
            break
    return first.strip()


def extract_model_answer(text: str, prompt_format: str = "chat") -> str:
    fmt = (prompt_format or "chat").strip().lower()
    if fmt == "completion":
        return extract_completion_answer(text)
    return extract_answer_text(text)


# =========================================================
# Config
# =========================================================

@dataclass
class StateMemoryConfig:
    batch_size: int = 8
    micro_batch_size: int = 0
    lr: float = 1e-4
    beta1: float = 0.95
    beta2: float = 0.99
    grad_clip: float = 0.5
    time_state_l2: float = 0.0

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 20
    mask_token0: bool = True

    use_rapid_sampling: bool = False
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    penalty_decay: float = 1.0

    stop_on_user: bool = True
    stop_on_newline: bool = True
    stop_check_every: int = 2
    stop_check_window: int = 64

    log_interval: int = 5
    eval_interval: int = 20
    save_interval: int = 50
    prompt_format: str = "chat"
    early_stop_acc: float = 0.0
    layerwise_lr_scheme: str = "flat"
    layerwise_lr_gamma: float = 1.0


# =========================================================
# Model helpers
# =========================================================

HEAD_SIZE = 64


def normalize_model_arg(model_arg: str) -> Tuple[str, str]:
    model_arg = model_arg.strip()
    if model_arg.endswith(".pth"):
        base = model_arg[:-4]
        pth = model_arg
    else:
        base = model_arg
        pth = model_arg + ".pth"
    if not os.path.isfile(pth) and os.path.isfile(base):
        pth = model_arg
        if pth.endswith(".pth"):
            base = pth[:-4]
    return base, pth


def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int, grad_cp: int = 0):
    from types import SimpleNamespace
    from rwkv7_trainable import RWKV7

    sd = _torch_load_weights(pth_path)

    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim_ffn = sd.get("blocks.0.ffn.key.weight", torch.zeros(n_embd * 4, n_embd)).shape[0]

    args = SimpleNamespace(
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_layer=n_layer,
        dim_att=n_embd,
        dim_ffn=dim_ffn,
        head_size_a=HEAD_SIZE,
        head_size_divisor=8,
        ctx_len=ctx_len,
        chunk_ctx=ctx_len,
        grad_cp=grad_cp,
        train_type="fullstate",
        peft="none",
        my_testing="x070",
    )

    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(device).to(torch.bfloat16)
    return model, args


def load_infer_model_fp16(base_name_no_pth: str, device: str = "cuda"):
    import types
    from rwkv7_fp16 import RWKV_x070

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    model = RWKV_x070(args)
    return model, args


def freeze_except_time_state(model: torch.nn.Module) -> int:
    cnt = 0
    for n, p in model.named_parameters():
        if "time_state" in n or "ts_state" in n:
            p.data = p.data.float()
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    return cnt


def save_time_state_only(model: torch.nn.Module, path: str):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if ("time_state" in n or "ts_state" in n)}
    torch.save(sd, path)


def load_time_state_only(model: torch.nn.Module, path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    sd = _torch_load_weights(path)
    if "full_state" in sd and isinstance(sd["full_state"], dict):
        sd = sd["full_state"]
    elif "time_state" in sd and isinstance(sd["time_state"], dict):
        sd = sd["time_state"]
    hit = 0
    for n, p in model.named_parameters():
        if n in sd:
            p.data.copy_(sd[n].to(p.device).to(p.dtype))
            hit += 1
    return hit > 0


def build_prefix_text_from_data(rows: List[Dict[str, Any]], prompt_format: str = "completion") -> str:
    fmt = (prompt_format or "completion").strip().lower()
    lines: List[str] = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        if not problem:
            continue
        if fmt == "completion":
            lines.append(f"Question: {problem} Answer: {solution}")
        else:
            lines.append(
                "你是一个问答助手。请直接回答，不要解释，不要重复问题。\n"
                f"User: {problem}\n"
                f"Assistant: {solution}"
            )
    return "\n".join(lines).strip()


def runtime_state_to_time_state_dict(model: torch.nn.Module, runtime_state, batch_idx: int = 0) -> Dict[str, torch.Tensor]:
    if runtime_state is None or len(runtime_state) != 2:
        raise ValueError("runtime_state must be a 2-item sequence: [token_state, time_state]")
    token_state, time_state = runtime_state
    if token_state is None or time_state is None:
        raise ValueError("runtime_state entries must not be None")
    if token_state.dim() != 4 or time_state.dim() != 5:
        raise ValueError(
            f"Unexpected runtime_state shapes: token_state={tuple(token_state.shape)} time_state={tuple(time_state.shape)}"
        )
    if batch_idx < 0 or batch_idx >= token_state.shape[2] or batch_idx >= time_state.shape[1]:
        raise IndexError(
            f"batch_idx={batch_idx} out of range for token_state B={token_state.shape[2]} time_state B={time_state.shape[1]}"
        )

    out: Dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.blocks):
        out[f"blocks.{i}.att.time_state"] = time_state[i, batch_idx].detach().float().cpu()
        if hasattr(block.att, "ts_state"):
            out[f"blocks.{i}.att.ts_state"] = token_state[i, 0, batch_idx].detach().float().cpu()
        if hasattr(block.ffn, "ts_state"):
            out[f"blocks.{i}.ffn.ts_state"] = token_state[i, 1, batch_idx].detach().float().cpu()
    return out


def apply_runtime_state_to_model(model: torch.nn.Module, runtime_state, batch_idx: int = 0) -> int:
    sd = runtime_state_to_time_state_dict(model, runtime_state, batch_idx=batch_idx)
    named_params = dict(model.named_parameters())
    hit = 0
    for n, v in sd.items():
        p = named_params.get(n)
        if p is None:
            continue
        p.data.copy_(v.to(device=p.device, dtype=p.dtype))
        hit += 1
    return hit


def save_runtime_state_as_time_state(
    model: torch.nn.Module,
    runtime_state,
    path: str,
    batch_idx: int = 0,
    meta: Optional[Dict[str, Any]] = None,
):
    payload = runtime_state_to_time_state_dict(model, runtime_state, batch_idx=batch_idx)
    if meta:
        payload = {
            "time_state": payload,
            "meta": meta,
        }
    torch.save(payload, path)


# =========================================================
# FP16 batched inference
# =========================================================

class FP16BatchInference:
    def __init__(self, infer_model, train_model, encode_fn, decode_fn, device: str, cfg: StateMemoryConfig):
        self.infer_model = infer_model
        self.train_model = train_model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.cfg = cfg
        self.vocab_size = infer_model.args.vocab_size
        self.sample_kernel = None

        if cfg.use_rapid_sampling:
            from torch.utils.cpp_extension import load
            print("Loading Rapid-Sampling kernel...", flush=True)
            self.sample_kernel = load(
                name="sample",
                sources=[f"{RAPID_SAMPLING_DIR}/sampling.cpp", f"{RAPID_SAMPLING_DIR}/sampling.cu"],
                extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
                verbose=False,
            )
            print("Rapid-Sampling kernel loaded.", flush=True)

    def init_state_with_time_state(self, B: int):
        set_current_cuda_device(self.device)
        infer_device = self.device
        args = self.infer_model.args
        state = [None, None]
        state[0] = torch.zeros((args.n_layer, 2, B, args.n_embd), dtype=torch.half, requires_grad=False, device=infer_device)
        state[1] = torch.zeros(
            (args.n_layer, B, args.n_embd // args.head_size, args.head_size, args.head_size),
            dtype=torch.half,
            requires_grad=False,
            device=infer_device,
        )
        for i, block in enumerate(self.train_model.blocks):
            ts = block.att.time_state
            att_ts = block.att.ts_state if hasattr(block.att, "ts_state") else None
            ffn_ts = block.ffn.ts_state if hasattr(block.ffn, "ts_state") else None
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone().to(device=infer_device, dtype=torch.half)
            if att_ts is not None:
                state[0][i, 0] = att_ts.to(device=infer_device, dtype=torch.half).unsqueeze(0).expand(B, -1).clone()
            if ffn_ts is not None:
                state[0][i, 1] = ffn_ts.to(device=infer_device, dtype=torch.half).unsqueeze(0).expand(B, -1).clone()
        return state

    @torch.no_grad()
    def prime_prompts(self, prompt_tokens_list: List[List[int]]):
        set_current_cuda_device(self.device)
        B = len(prompt_tokens_list)
        state = self.init_state_with_time_state(B)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_user: bool,
        stop_on_newline: bool,
        stop_check_every: int,
        stop_check_window: int,
        stop_on_boxed: bool = False,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        penalty_decay: Optional[float] = None,
        rng_seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:
        set_current_cuda_device(self.device)
        presence_penalty = presence_penalty if presence_penalty is not None else self.cfg.presence_penalty
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.cfg.repetition_penalty
        penalty_decay = penalty_decay if penalty_decay is not None else self.cfg.penalty_decay

        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []

        last_logits, state = self.prime_prompts(prompt_tokens_list)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]

        penalties = None
        rand_states = None
        if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
            rand_seed = int(time.time()) if rng_seed is None else int(rng_seed)
            rand_states = self.sample_kernel.setup_rand(rand_seed, B)
            vocab_padded = ((self.vocab_size + 3) // 4) * 4
            penalties = torch.zeros(B, vocab_padded, device=self.device, dtype=torch.float32)

        def has_balanced_boxed(text: str) -> bool:
            s = str(text or "")
            key = r"\boxed{"
            idx = s.rfind(key)
            if idx < 0:
                return False
            i = idx + len(key) - 1
            if i >= len(s) or s[i] != "{":
                return False
            depth = 0
            j = i
            while j < len(s):
                ch = s[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        inner = s[i + 1:j].strip()
                        return inner not in {"", "...", "…", "．．．"}
                j += 1
            return False

        def sample_next(logits_2d: torch.Tensor) -> torch.Tensor:
            if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
                logits_float = logits_2d.float()
                if logits_float.size(-1) % 4 != 0:
                    pad_size = 4 - (logits_float.size(-1) % 4)
                    logits_float = F.pad(logits_float, (0, pad_size), value=-1e30)
                return self.sample_kernel.batch_sampling_repetition_temperature_topk_topp(
                    logits_float,
                    penalties,
                    rand_states,
                    presence_penalty,
                    frequency_penalty,
                    penalty_decay,
                    temperature,
                    top_k,
                    top_p,
                )

            if temperature <= 0:
                return torch.argmax(logits_2d, dim=-1)

            x = logits_2d.float() / float(temperature)
            V = x.size(-1)
            if self.cfg.mask_token0:
                x[:, 0] = -1e30

            k_cap = 0
            if top_k and top_k > 0:
                k_cap = int(min(top_k, V))
            elif top_p and 0.0 < top_p < 1.0:
                k_cap = int(min(2048, V))

            if k_cap > 0:
                topv, topi = torch.topk(x, k=k_cap, dim=-1)
                if top_p and 0.0 < top_p < 1.0:
                    probs = F.softmax(topv, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    keep = cdf <= float(top_p)
                    keep[:, 0] = True
                    topv = topv.masked_fill(~keep, -1e30)
                probs = F.softmax(topv, dim=-1)
                pick = torch.multinomial(probs, 1).squeeze(-1)
                return topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1)

            probs = F.softmax(x, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        for t in range(max_new_tokens):
            if not bool(active.any().item()):
                break

            logits = last_logits
            token_ids = sample_next(logits).long()
            logp_all = F.log_softmax(logits.float(), dim=-1)
            picked_logp = logp_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()
            for i in range(B):
                if not active[i]:
                    continue
                comp_tokens[i].append(int(tok_cpu[i]))
                old_logps[i].append(float(lp_cpu[i]))

            stop_tokens = {0, 24281}
            for i in range(B):
                if active[i] and int(tok_cpu[i]) in stop_tokens:
                    active[i] = False

            if (stop_on_user or stop_on_newline or stop_on_boxed) and (t % max(1, stop_check_every) == 0):
                for i in range(B):
                    if not active[i]:
                        continue
                    window = comp_tokens[i][-stop_check_window:] if stop_check_window > 0 else comp_tokens[i]
                    s = self.decode(window)
                    if stop_on_boxed and has_balanced_boxed(s):
                        active[i] = False
                        continue
                    if stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[i] = False
                        continue
                    if stop_on_newline and ("\n" in s.strip("\r")):
                        active[i] = False
                        continue

            step_tokens_batch = [int(x) for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        for i in range(B):
            if bool(active[i].item()):
                truncated[i] = True

        if penalties is not None:
            del penalties
        if rand_states is not None:
            del rand_states
        torch.cuda.empty_cache()

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_tokens, old_logps, comp_text, truncated


# =========================================================
# Trainer
# =========================================================

class StateMemoryTrainer:
    def _parse_block_index(self, name: str) -> Optional[int]:
        m = re.match(r"blocks\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def _layer_lr_scale(self, name: str) -> float:
        scheme = str(self.cfg.layerwise_lr_scheme or "flat").strip().lower()
        gamma = float(self.cfg.layerwise_lr_gamma)
        if scheme == "flat" or gamma == 1.0:
            return 1.0
        layer_idx = self._parse_block_index(name)
        if layer_idx is None:
            return 1.0
        n_layer = int(getattr(self.model.args, "n_layer", 0) or 0)
        if n_layer <= 1:
            return 1.0
        if scheme == "top_down":
            depth = (n_layer - 1) - layer_idx
            return gamma ** depth
        if scheme == "bottom_up":
            depth = layer_idx
            return gamma ** depth
        return 1.0

    def _build_optimizer(self):
        grouped = {}
        group_meta = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            scale = self._layer_lr_scale(name)
            lr = float(self.cfg.lr) * float(scale)
            key = round(lr, 12)
            grouped.setdefault(key, []).append(param)
            meta = group_meta.setdefault(key, {
                "lr": lr,
                "num_tensors": 0,
                "numel": 0,
                "sample_names": [],
            })
            meta["num_tensors"] += 1
            meta["numel"] += param.numel()
            if len(meta["sample_names"]) < 3:
                meta["sample_names"].append(name)

        if not grouped:
            raise RuntimeError("No trainable params found; expected time_state / ts_state only.")

        param_groups = [{"params": params, "lr": group_meta[key]["lr"]} for key, params in sorted(grouped.items(), key=lambda x: x[0])]
        opt = torch.optim.Adam(
            param_groups,
            betas=(float(self.cfg.beta1), float(self.cfg.beta2)),
            eps=1e-18,
            weight_decay=0.0,
        )
        return opt, [group_meta[key] for key in sorted(group_meta)]

    def __init__(
        self,
        train_model,
        infer_engine: FP16BatchInference,
        encode_fn,
        decode_fn,
        out_dir: str,
        device: str,
        cfg: StateMemoryConfig,
        seed: int = 42,
    ):
        self.model = train_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = random.Random(seed)

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")
        self.pred_path = os.path.join(out_dir, "predictions.jsonl")
        self.stage_path = os.path.join(out_dir, "stage_summary.jsonl")

        self.opt, self.opt_group_meta = self._build_optimizer()

        self._ts_init: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" in n or "ts_state" in n:
                    self._ts_init[n] = p.detach().clone()

        for meta in self.opt_group_meta:
            print(
                f"[opt] lr={meta['lr']:.8g} num_tensors={meta['num_tensors']} numel={meta['numel']} sample={meta['sample_names']}",
                flush=True,
            )

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _sample_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            raise RuntimeError("empty training data")
        if len(data) <= self.cfg.batch_size:
            batch = list(data)
            self.rng.shuffle(batch)
            return batch
        return self.rng.sample(data, self.cfg.batch_size)

    def _encode_example(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = build_item_prompt(item, prompt_format=self.cfg.prompt_format)
        answer = (item["solution"] or "").strip()
        target_text = f" {answer}\n"

        prompt_ids = self.encode(prompt)
        target_ids = self.encode(target_text)
        full_ids = prompt_ids + target_ids

        if len(full_ids) < 2:
            return None

        x = full_ids[:-1]
        y = full_ids[1:]
        mask = [0] * max(0, len(prompt_ids) - 1) + [1] * len(target_ids)
        if len(mask) != len(x):
            return None

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "problem": item["problem"],
            "solution": answer,
            "fact_id": item.get("fact_id"),
            "prompt": prompt,
        }

    def _build_tensors_from_encoded(self, encoded: List[Dict[str, Any]]):
        if not encoded:
            raise RuntimeError("Failed to build a valid batch.")

        max_len = max(len(x["x"]) for x in encoded)
        B = len(encoded)

        x = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        y = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        m = torch.zeros((B, max_len), dtype=torch.float32, device=self.device)

        for i, ex in enumerate(encoded):
            n = len(ex["x"])
            x[i, :n] = torch.tensor(ex["x"], dtype=torch.long, device=self.device)
            y[i, :n] = torch.tensor(ex["y"], dtype=torch.long, device=self.device)
            m[i, :n] = torch.tensor(ex["mask"], dtype=torch.float32, device=self.device)
        return x, y, m, encoded

    def _build_batch_tensors(self, batch: List[Dict[str, Any]]):
        encoded = []
        for item in batch:
            ex = self._encode_example(item)
            if ex is not None:
                encoded.append(ex)
        return self._build_tensors_from_encoded(encoded)

    def _time_state_reg(self) -> torch.Tensor:
        if self.cfg.time_state_l2 <= 0:
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._ts_init:
                reg = reg + (p.float() - self._ts_init[n].float().to(p.device)).pow(2).mean()
        return reg

    def train_step(self, train_data: List[Dict[str, Any]]) -> Dict[str, float]:
        set_current_cuda_device(self.device)
        self.model.train()
        batch = self._sample_batch(train_data)
        _, _, _, encoded = self._build_batch_tensors(batch)

        micro_bs = int(self.cfg.micro_batch_size) if int(self.cfg.micro_batch_size) > 0 else len(encoded)
        micro_bs = max(1, min(micro_bs, len(encoded)))
        denom_value = float(sum(sum(ex["mask"]) for ex in encoded))
        denom = torch.tensor(max(1.0, denom_value), dtype=torch.float32, device=self.device)

        ce_num = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        correct_num = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.opt.zero_grad(set_to_none=True)
        for start in range(0, len(encoded), micro_bs):
            chunk = encoded[start:start + micro_bs]
            x, y, m, _ = self._build_tensors_from_encoded(chunk)

            logits = self.model(x)
            if not torch.is_tensor(logits):
                raise RuntimeError("train model forward did not return logits tensor")

            vocab = logits.size(-1)
            loss_tok = F.cross_entropy(
                logits.float().reshape(-1, vocab),
                y.reshape(-1),
                reduction="none",
            ).reshape_as(m)

            ce_num_chunk = (loss_tok * m).sum()
            (ce_num_chunk / denom).backward()
            ce_num = ce_num + ce_num_chunk.detach()

            with torch.no_grad():
                pred = torch.argmax(logits.float(), dim=-1)
                correct_num = correct_num + (((pred == y).float() * m).sum())

        reg_loss = self._time_state_reg()
        if float(self.cfg.time_state_l2) > 0:
            (float(self.cfg.time_state_l2) * reg_loss).backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_([
            p for p in self.model.parameters() if p.requires_grad
        ], self.cfg.grad_clip).item())
        self.opt.step()

        ce_loss = ce_num / denom
        tok_acc = correct_num / denom
        total_loss = ce_loss + float(self.cfg.time_state_l2) * reg_loss

        return {
            "loss": float(total_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "token_acc": float(tok_acc.item()),
            "grad_norm": grad_norm,
            "batch_size": float(len(encoded)),
        }

    @torch.no_grad()
    def evaluate(self, eval_data: List[Dict[str, Any]], step: int, stage_name: str, tag: str) -> Dict[str, Any]:
        set_current_cuda_device(self.infer.device)
        self.model.eval()
        prompt_texts = [build_item_prompt(item, prompt_format=self.cfg.prompt_format) for item in eval_data]
        prompts = [self.encode(text) for text in prompt_texts]
        comp_tokens, _, comp_texts, truncated = self.infer.generate_group_parallel(
            prompt_tokens_list=prompts,
            group_size=1,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            stop_on_user=self.cfg.stop_on_user,
            stop_on_newline=self.cfg.stop_on_newline,
            stop_check_every=self.cfg.stop_check_every,
            stop_check_window=self.cfg.stop_check_window,
        )

        correct = 0
        rows = []
        for idx, item in enumerate(eval_data):
            raw = comp_texts[idx] if idx < len(comp_texts) else ""
            pred = extract_model_answer(raw, prompt_format=self.cfg.prompt_format)
            ok = exact_match(pred, item["solution"])
            if ok:
                correct += 1
            row = {
                "step": step,
                "stage": stage_name,
                "tag": tag,
                "fact_id": item.get("fact_id"),
                "problem": item["problem"],
                "prompt": prompt_texts[idx] if idx < len(prompt_texts) else None,
                "gt": item["solution"],
                "pred": pred,
                "raw": raw,
                "full_output": raw,
                "full_output_repr": repr(raw),
                "generated_token_ids": comp_tokens[idx] if idx < len(comp_tokens) else None,
                "generated_token_count": len(comp_tokens[idx]) if idx < len(comp_tokens) else 0,
                "correct": ok,
                "truncated": bool(truncated[idx]) if idx < len(truncated) else None,
            }
            rows.append(row)
            append_jsonl(self.pred_path, row)

        total = max(1, len(eval_data))
        acc = correct / total
        summary = {
            "step": step,
            "stage": stage_name,
            "tag": tag,
            "total": len(eval_data),
            "correct": correct,
            "accuracy": acc,
        }
        append_jsonl(self.eval_path, summary)

        sample_preview = rows[0] if rows else {}
        self._log(
            f"[eval] step={step} stage={stage_name} tag={tag} acc={acc:.4f} "
            f"sample_gt={sample_preview.get('gt', '')!r} sample_pred={sample_preview.get('pred', '')!r}"
        )
        return summary

    def save_checkpoint(self, step: int, stage_name: str):
        path = os.path.join(self.out_dir, f"time_state_{stage_name}_step{step}.pth")
        latest = os.path.join(self.out_dir, "latest_time_state.pth")
        save_time_state_only(self.model, path)
        save_time_state_only(self.model, latest)
        self._log(f"[save] {path}")

    def train_curriculum(self, stages: List[Dict[str, Any]], steps_per_stage: int):
        global_step = 0

        for stage in stages:
            fact_count = int(stage["fact_count"])
            stage_name = f"k{fact_count}"
            train_data = stage["train"]
            eval_data = stage["eval"]

            self._log(
                f"[stage] start {stage_name} train={len(train_data)} eval={len(eval_data)} facts={fact_count}"
            )
            append_jsonl(self.stage_path, {
                "event": "stage_start",
                "stage": stage_name,
                "fact_count": fact_count,
                "train_size": len(train_data),
                "eval_size": len(eval_data),
                "step": global_step,
            })

            self.evaluate(eval_data, step=global_step, stage_name=stage_name, tag="before_train")

            for local_step in range(1, steps_per_stage + 1):
                global_step += 1
                metrics = self.train_step(train_data)

                if global_step % self.cfg.log_interval == 0 or local_step == 1:
                    self._log(
                        f"[train] step={global_step} stage={stage_name} "
                        f"loss={metrics['loss']:.4f} ce={metrics['ce_loss']:.4f} "
                        f"tok_acc={metrics['token_acc']:.4f} grad={metrics['grad_norm']:.4f}"
                    )

                if global_step % self.cfg.eval_interval == 0:
                    summary = self.evaluate(eval_data, step=global_step, stage_name=stage_name, tag="during_train")
                    if float(self.cfg.early_stop_acc) > 0 and float(summary.get("accuracy", 0.0)) >= float(self.cfg.early_stop_acc):
                        self._log(
                            f"[early_stop] step={global_step} stage={stage_name} acc={summary['accuracy']:.4f} threshold={self.cfg.early_stop_acc:.4f}"
                        )
                        self.save_checkpoint(global_step, stage_name)
                        append_jsonl(self.stage_path, {
                            "event": "early_stop",
                            "stage": stage_name,
                            "fact_count": fact_count,
                            "step": global_step,
                            **summary,
                        })
                        return

                if global_step % self.cfg.save_interval == 0:
                    self.save_checkpoint(global_step, stage_name)

            final_summary = self.evaluate(eval_data, step=global_step, stage_name=stage_name, tag="after_train")
            self.save_checkpoint(global_step, stage_name)

            append_jsonl(self.stage_path, {
                "event": "stage_end",
                "stage": stage_name,
                "fact_count": fact_count,
                "step": global_step,
                **final_summary,
            })


# =========================================================
# Sanity check
# =========================================================

@torch.no_grad()
def sanity_check_train_vs_fp16(train_model, infer_model, infer_engine, encode, decode, train_device="cuda:0", prompt_format="chat"):
    set_current_cuda_device(train_device)
    prompt = build_prompt("2026年云桥市记忆节的吉祥物叫什么？", prompt_format=prompt_format)
    ids = encode(prompt)
    if not ids:
        raise RuntimeError("encode(prompt) returned empty")

    t = torch.tensor([ids], device=train_device, dtype=torch.long)
    logits_train = train_model(t)[0, -1].float().cpu()

    state = infer_engine.init_state_with_time_state(B=1)
    out2 = infer_model.forward_batch([ids], state)
    if torch.is_tensor(out2) and out2.dim() == 3:
        out2 = out2[:, -1, :]
    logits_infer = out2[0].float().cpu()

    top_train = int(torch.argmax(logits_train).item())
    top_infer = int(torch.argmax(logits_infer).item())

    print("[sanity] top1_train =", top_train, "->", repr(decode([top_train])), flush=True)
    print("[sanity] top1_infer =", top_infer, "->", repr(decode([top_infer])), flush=True)

    if top_train != top_infer:
        top5_train = torch.topk(logits_train, 5).indices.tolist()
        top5_infer = torch.topk(logits_infer, 5).indices.tolist()
        overlap = len(set(top5_train) & set(top5_infer))
        print("[sanity] WARNING: train vs infer top1 mismatch - may be due to precision diff", flush=True)
        print(f"[sanity] top5_train = {top5_train}", flush=True)
        print(f"[sanity] top5_infer = {top5_infer}", flush=True)
        print(f"[sanity] top5 overlap = {overlap}/5", flush=True)
        if overlap < 3:
            raise RuntimeError("SANITY FAIL: train vs infer top5 has <3 overlap - check model loading")


# =========================================================
# Main
# =========================================================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="rwkv7-g1c-2.9b-20251231-ctx8192", help="model path")
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--out_dir", type=str, default=f"out_state_memory_{now_str()}")
    ap.add_argument("--train_jsonl", type=str, default="", help="optional external train jsonl/json")
    ap.add_argument("--eval_data", type=str, default="", help="optional external eval jsonl/json")
    ap.add_argument("--state_init", type=str, default="")
    ap.add_argument("--prefix_init_text", type=str, default="", help="txt file containing explicit prefix to prefill into state")
    ap.add_argument("--prefix_init_dataset", type=str, default="", help="json/jsonl QA data used to build explicit prefix lines")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1)
    ap.add_argument("--train_cuda_idx", type=int, default=1)
    ap.add_argument("--infer_cuda_idx", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch_size", type=int, default=0)
    ap.add_argument("--steps_per_stage", type=int, default=80)
    ap.add_argument("--fact_schedule", type=str, default="1,2,4,8")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.95)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--time_state_l2", type=float, default=0.0)

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--prompt_format", type=str, default="chat", choices=["chat", "completion"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--use_rapid_sampling", action="store_true")

    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--eval_interval", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--early_stop_acc", type=float, default=0.0)
    ap.add_argument("--layerwise_lr_scheme", type=str, default="flat", choices=["flat", "top_down", "bottom_up"])
    ap.add_argument("--layerwise_lr_gamma", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = f"cuda:{args.train_cuda_idx}"
    infer_device = f"cuda:{args.infer_cuda_idx}"
    n_cuda = torch.cuda.device_count()
    print(f"[GPU] detected {n_cuda} visible GPUs. train={device}, infer={infer_device}", flush=True)

    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_jsonl:
        train_data = load_data(args.train_jsonl)
        eval_data = load_data(args.eval_data) if args.eval_data else train_data
        stages = [{
            "stage_idx": 1,
            "fact_count": -1,
            "facts": [],
            "train": train_data,
            "eval": eval_data,
        }]
    else:
        fact_bank = build_synthetic_2026_facts()
        schedule = parse_fact_schedule(args.fact_schedule, max_facts=len(fact_bank))
        stages = build_curriculum(schedule, fact_bank)
        dump_curriculum_data(args.out_dir, stages)
        print(f"[data] synthetic 2026 curriculum stages: {schedule}", flush=True)

    from utils import TRIE_TOKENIZER

    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)

    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except TypeError:
            pass
        try:
            return tok.decode(ids)
        except UnicodeDecodeError:
            try:
                b = tok.decodeBytes(ids)
                return b.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(int(x) % 256) for x in ids)

    decode = safe_decode

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    print(f"Loading model: {pth_path}", flush=True)
    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len), grad_cp=int(args.grad_cp))
    infer_model, _ = load_infer_model_fp16(base_name, device=infer_device)

    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError("No trainable time_state found. Check model weights / naming.")
    print(f"Trainable parameters (time_state only): {trainable}", flush=True)

    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"[state_init] loaded={ok} from {args.state_init}", flush=True)

    cfg = StateMemoryConfig(
        batch_size=int(args.batch_size),
        micro_batch_size=int(args.micro_batch_size),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        grad_clip=float(args.grad_clip),
        time_state_l2=float(args.time_state_l2),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        use_rapid_sampling=bool(args.use_rapid_sampling),
        log_interval=int(args.log_interval),
        eval_interval=int(args.eval_interval),
        save_interval=int(args.save_interval),
        prompt_format=str(args.prompt_format),
        early_stop_acc=float(args.early_stop_acc),
        layerwise_lr_scheme=str(args.layerwise_lr_scheme),
        layerwise_lr_gamma=float(args.layerwise_lr_gamma),
    )

    if cfg.prompt_format == "completion":
        cfg.stop_on_user = False
        cfg.stop_on_newline = True

    infer_engine = FP16BatchInference(infer_model, train_model, encode, decode, device=infer_device, cfg=cfg)

    prefix_init_text = ""
    if args.prefix_init_text:
        with open(args.prefix_init_text, "r", encoding="utf-8") as f:
            prefix_init_text = f.read().strip()
    elif args.prefix_init_dataset:
        prefix_init_text = build_prefix_text_from_data(load_data(args.prefix_init_dataset), prompt_format="completion")
    if prefix_init_text:
        prefix_init_text = prefix_init_text.strip() + "\n\n"
        with torch.no_grad():
            _, runtime_state = infer_engine.prime_prompts([encode(prefix_init_text)])
            copied = apply_runtime_state_to_model(train_model, runtime_state, batch_idx=0)
        print(
            f"[prefix_init] copied={copied} tensors from explicit prefix "
            f"(chars={len(prefix_init_text)} tokens={len(encode(prefix_init_text))})",
            flush=True,
        )

    sanity_check_train_vs_fp16(
        train_model,
        infer_model,
        infer_engine,
        encode,
        decode,
        train_device=device,
        prompt_format=cfg.prompt_format,
    )

    trainer = StateMemoryTrainer(
        train_model=train_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
    )
    trainer.train_curriculum(stages, steps_per_stage=int(args.steps_per_stage))


if __name__ == "__main__":
    main()
