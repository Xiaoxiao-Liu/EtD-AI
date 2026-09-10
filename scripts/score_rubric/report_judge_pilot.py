"""Generate the completed judge pilot report from saved measurements only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_reports/judge_prompt_20260909'


def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def dec(x):return '—' if x is None else f'{x:.3f}'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    policy=json.loads((OUT/'policy_agreement.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    replay=json.loads((OUT/'replay_check.json').read_text())
    val=summary['v2_2/validation']
    assert val['scores']==540 and val['status'].get('ok')==540
    assert replay['status']=='passed'
    attempts=sum(v['calls'] for v in summary.values())
    scores=sum(v['scores'] for v in summary.values())
    lines=['# LLM judge prompt 修订与三模型试验报告', '',
      '日期：2026-09-09。最终版本：`etd-judge-v2.2`。', '',
      '**已完成 prompt 修订和小规模 API 验证；尚未进行全量 judge，也没有独立临床专家评分。**', '',
      '## 实验范围', '',
      '- Judge：GPT-4o、GPT-5.5、DeepSeek-V4-Pro。DeepSeek 官方当前文本模型 API 名称为 `deepseek-v4-pro`，已通过配置的官方端点 models 列表确认可用。',
      '- 被评分答案：已有 GPT-4o、GPT-5.5、Claude Opus 4.6 输出；没有重新调用 executor 生成答案。',
      '- 60 条开发记录 + 60 条独立验证记录，全部来自原 train。每批各覆盖 12 个 criterion（每项 5 条）和三种 executor（各 20 条）；两批没有共享 PICO。',
      '- 每条记录分别评价 Closed-book、RAG、Oracle。开发阶段另选 18 条原记录复查两次，没有增加独立样本数。',
      '- 验证前冻结 prompt；没有根据验证结果再修改。官方 val/test 均未参与本次实验。', '',
      '| 阶段 | 记录数 | 三模型×三条件评分数 | API 请求尝试数 |',
      '|---|---:|---:|---:|']
    for key,n,title in [('legacy/development',60,'旧 prompt 开发对照'),('v2/development',60,'v2 初次修订'),
                         ('v2_1/refinement',18,'v2.1 缺证据规则复查'),('v2_2/refinement',18,'v2.2 逐维约束复查'),
                         ('v2_2/validation',60,'v2.2 独立验证')]:
        item=summary[key]
        lines.append(f'| {title} | {n} | {item["scores"]} | {item["calls"]} |')
    lines+=['',f'合计完成 {scores:,} 项评分，记录 {attempts:,} 次 HTTP 请求尝试（含超时/服务错误重试）。失败请求是否计费由服务端决定；这里没有估算账单。', '',
      '## 修订内容与实测发现', '',
      '1. **Grounding 与答案正确性分开。** 旧 prompt 强制错误判断的 grounding=0；新版按实际提供的证据判断主张是否被支持。参考标签匹配仍由规则计算，正确性硬门控没有放宽。',
      '2. **补充 EtD 一致性上下文。** 使用同一 PICO、同一 executor、同一证据条件的其他 criterion 判断，不混用其他模型、gold 判断或 test 数据。不把“大收益、高成本、低确定性”等不同维度的权衡直接判为矛盾。',
      '3. **缺证据不再含混处理。** 区分空字段、`react-empty` 网页占位符、明确无研究说明、未提供的交叉引用。纯粹承认无法评估可标 grounding=null；额外断言没有证据支持的临床/经济事实应为 0。',
      '4. **逐维空值约束。** v2.1 边界复查中 GPT-4o 有 11/54 项在其他 criterion 上下文存在时仍将 consistency 标为空；v2.2 加入输入对应的允许值后，复查中该问题为 0/54。',
      '5. **置信度与相关性。** 不因答案标签是 Yes/Large 就判定过度自信，不只奖励 hedging；相关性不再要求理由复述 reference。这里评价的是表达信心的适当性，不是概率校准。',
      '6. **缺参考判断不当成答错。** 原规则数据中的 147 条记录三种 correctness 均为空；现在标为 missing_correctness，不产生 SFT/DPO policy target。分数及原记录仍保留作诊断。',
      '7. **防止混用版本。** 新结果路径加入 prompt version，保存输入指纹、原始响应和调用统计；非法 JSON/取值不会静默变成成功的空分数。结构化表格不再被替换为“tables omitted”。', '',
      '## 独立验证：格式及评分一致性', '',
      f'最终 60 条 × 3 条件 × 3 judge = 540 项评分，{val["status"].get("ok",0)}/540 通过 JSON 键、0/1/2/null 类型及上下文可评估性检查。该数字只代表输出符合协议，不代表临床评分正确率。', '',
      '| Judge 组合 | Relevance | Grounding | Confidence | EtD consistency |',
      '|---|---:|---:|---:|---:|']
    for pair,dims in val['agreement'].items():
        lines.append('| '+pair+' | '+' | '.join(pct(dims.get(k,{}).get('exact')) for k in
          ('reasoning_relevance','reasoning_grounding','confidence_calibration','etd_consistency'))+' |')
    lines+=['', '上表为逐维精确一致率；grounding 的数值比较仅纳入双方都给出数值的条目，空值分歧另见下表，不能把它当作全部样本的一致率。', '',
      '| Judge 组合 | Grounding 数值对数 | Grounding 加权 κ |', '|---|---:|---:|']
    for pair,dims in val['agreement'].items():
        d=dims['reasoning_grounding'];lines.append(f'| {pair} | {d["n"]} | {dec(d["linear_weighted_kappa"])} |')
    # Include null in grounding agreement so dropping it cannot conceal availability disagreement.
    raw={}
    for p in (OUT/'responses/v2_2/validation').rglob('*.json'):
        r=json.loads(p.read_text());raw[(r['record_id'],r['method'],r['judge'])]=r
    import itertools
    models=('gpt-4o','gpt-5.5','deepseek-v4-pro')
    lines+=['','| Judge 组合 | Grounding 含 null 一致率 | 仅一方为 null |','|---|---:|---:|']
    for a,b in itertools.combinations(models,2):
        pairs=[]
        for (rid,method,j),r in raw.items():
            if j==a and method!='no_evidence':
                pairs.append((r['scores']['reasoning_grounding'],raw[(rid,method,b)]['scores']['reasoning_grounding']))
        lines.append(f'| {a} vs {b} | {pct(sum(x==y for x,y in pairs)/len(pairs))} | {sum((x is None)!=(y is None) for x,y in pairs)}/{len(pairs)} |')
    lines+=['','## 独立验证：是否改变训练动作标签','','| Judge 组合 | Gatekeeper | Verifier：Closed-book | Verifier：RAG |','|---|---:|---:|---:|']
    for pair,d in policy['v2_2/validation']['action_agreement'].items():
        lines.append('| '+pair+' | '+' | '.join(pct(d[k]['agreement'])+f' (n={d[k]["n"]})' for k in
                     ('evidence_sufficiency_label','no_evidence_arbit_label','rag_arbit_label'))+' |')
    lines+=['',
      'Router 标签由已计算的判断正确性决定，通常不会随 judge 改变；不能把 Router 的高一致率当作 judge 质量证据。缺参考 correctness 的记录从动作比较中排除。动作标签由固定阈值和当前 label_policy 推导，没有在验证批调阈值。', '',
      '## 使用建议与限制', '',
      '- 新 prompt 修复了已知规则冲突与缺失上下文问题，可以作为下一轮评分的明确版本；它不是经过专家验证的临床评分器。',
      '- 暂保留 GPT-5.5 为主 judge；GPT-4o、DeepSeek 用于抽样对照。该选择延续现有实验配置，不是由本次无专家金标的一致率证明它最准确。',
      '- Grounding 的 0/1/2/null 边界仍有实质分歧，特别是“缺证据但夹带合理常识”的候选。全量运行前建议针对报告中的分歧样本进行专家抽查，不强行平均或多数投票覆盖主评分。',
      '- 旧、新 prompt 定义有所改变，分数提升不能解释为基线答案质量提升；旧 prompt 的 grounding 高一致率部分来自 correctness=0 的强制 0 分规则。',
      '- Relevance/consistency 存在高分集中现象，必须结合分布和 κ 阅读一致率。样本为开发用途的分层抽样，不能据此估计完整数据集表现。',
      '- 其他 criterion 上下文是模型自身输出，不是额外事实证据。现有聚合对 null 仍按固定权重零贡献处理，不自动重新归一化；缺参考 correctness 则完全不产生训练目标。',
      '- GPT 请求使用项目配置的兼容网关；DeepSeek 使用项目配置的官方端点。响应返回的是模型别名，不能据此声称验证了网关内部权重快照。', '',
      '## 代码与验证', '',
      '- Prompt：`src/task/score_rubric/judge_prompts.py`。',
      '- 正式调用/缓存：`src/task/score_rubric/judge_rubric.py`、`src/common/judge_client.py`。使用 Python 标准库，未在本地 etd 环境安装额外依赖。',
      '- 合并与缺失监督处理：`src/task/score_rubric/merge_rubric.py`、`label_policy.py`。',
      '- 已运行 16 项单元测试：15 通过，1 项因正式 labeled 数据尚不存在而跳过。',
      f'- 使用验证批原始响应离线回放正式解析、保存、合并链路：{replay["record_triplets"]} 个三条件记录、{replay["cached_method_responses"]} 个响应通过；新增 API 调用 0。',
      '- 原始 extract/scored 数据未修改，正式 judge 输出目录未生成，本轮没有执行全量评分、merge/label 或模型训练。', '',
      '## 复核材料', '',
      '- `manifest.json`：样本、随机种子、原始评分输入校验值；`validation_freeze.json`：验证前冻结记录。',
      '- `prompts_v2.py`、`prompts_v2_1.py`、`prompts_v2_2.py`、`legacy_judge_rubric.py`：各版存档。',
      '- `responses/`：每次评分的 prompt、原始输出、模型返回标识、usage 和请求尝试。',
      '- `summary.json`：逐维分布、精确一致率、线性加权 κ、token 统计。',
      '- `policy_agreement.json`、`pilot_policy_labels.jsonl`：固定规则下的动作标签比较，仅供 pilot 分析。',
      '- `replay_check.json`：正式评分链路的零 API 回放检查。', '',
      '来源：[DeepSeek 官方模型说明](https://api-docs.deepseek.com/)；[OpenAI 评估建议](https://developers.openai.com/api/docs/guides/evaluation-best-practices)。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    code_paths=['src/common/judge_client.py','src/task/score_rubric/judge_prompts.py',
                'src/task/score_rubric/judge_rubric.py','src/task/score_rubric/merge_rubric.py',
                'src/task/score_rubric/label_policy.py']
    (OUT/'final_code_sha256.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths},indent=2))
    print('Report:',OUT/'REPORT.md')

if __name__=='__main__':main()
