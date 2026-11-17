from settings import ROOT,save,digest
import json,statistics,zipfile

def main():
    rows={arm:[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'eval'/arm).glob('*.json'))] for arm in ('original','complete')}
    assert all(len(v)==32 for v in rows.values())
    summary={}
    for arm,rs in rows.items():
        main=[r for r in rs if r['record']['kind']!='intent'];math=[r for r in rs if r['record']['kind']=='gsm8k'];intent=[r for r in rs if r['record']['kind']=='intent']
        summary[arm]=dict(outputs=32,main_outputs=len(main),raw_adjacent=sum(r['metrics']['raw_adjacent'] for r in main),
            opportunities=sum(r['metrics']['opportunities'] for r in main),
            raw_adjacent_answers=sum(r['metrics']['raw_adjacent']>0 for r in main),
            repeat_answers={k:sum(r['metrics']['spans'][k]>0 for r in main) for k in ('token','phrase','sentence')},
            repeat_positions={k:sum(r['metrics']['repeated_positions'][k] for r in main) for k in ('token','phrase','sentence')},
            mean_seconds=statistics.mean(r['seconds'] for r in main),mean_tokens=statistics.mean(len(r['tokens']) for r in main),
            budget_answers=sum(len(r['tokens'])==192 for r in main),empty_answers=sum(not r['tokens'] for r in main),
            math_correct=sum(r['correct'] for r in math),math_total=len(math),
            intent_correct=sum(r['intent_correct'] for r in intent),intent_total=len(intent),
            actions=sum(len(r['actions']) for r in rs),changed=sum(a['changed'] for r in rs for a in r['actions']),
            action_answers=sum(bool(r['actions']) for r in rs),
            ar_reverified_actions=sum(a.get('ar_choice_committed',False) for r in rs for a in r['actions']))
    save('summary.json',summary)
    by={arm:{r['record']['id']:r for r in rs} for arm,rs in rows.items()}
    cases=[]
    for ident,a in by['original'].items():
        b=by['complete'][ident]
        cases.append(dict(id=ident,prompt=a['record']['prompt'],kind=a['record']['kind'],
            original=a['text'],complete=b['text'],original_metrics=a['metrics'],complete_metrics=b['metrics'],
            actions=b['actions'],original_correct=a.get('correct'),complete_correct=b.get('correct')))
    save('paired-cases.json',cases)
    a,b=summary['original'],summary['complete']
    report=['# 多尺度自由生成轨迹＋AR修复：小规模原始模型对比','',
        '32条配对任务：16通用、12数学、4指定重复控制。主表排除4条指定重复。单种子、96次更新；已观察过的提示集，不是盲测。','',
        '|指标|原始模型|完整方法|','|---|---:|---:|',
        f'|原始相邻同ID事件/机会|{a["raw_adjacent"]}/{a["opportunities"]}|{b["raw_adjacent"]}/{b["opportunities"]}|',
        f'|相邻同ID率|{100*a["raw_adjacent"]/max(1,a["opportunities"]):.3f}%|{100*b["raw_adjacent"]/max(1,b["opportunities"]):.3f}%|']
    for k,name in [('token','异常相邻token'),('phrase','短语'),('sentence','句子/句式')]:
        report.append(f'|{name}重复回答数/28|{a["repeat_answers"][k]}|{b["repeat_answers"][k]}|')
    report += [f'|数学答案抽取匹配/12|{a["math_correct"]}|{b["math_correct"]}|',
        f'|指定重复严格匹配/4|{a["intent_correct"]}|{b["intent_correct"]}|',
        f'|平均token数|{a["mean_tokens"]:.1f}|{b["mean_tokens"]:.1f}|',
        f'|平均秒|{a["mean_seconds"]:.2f}|{b["mean_seconds"]:.2f}|',
        f'|达到192 token预算的回答数|{a["budget_answers"]}|{b["budget_answers"]}|','',
        f'正式推理实际修复 {b["actions"]} 次，目标token改变 {b["changed"]} 次，涉及 {b["action_answers"]} 条回答；明确提交AR裁决结果 {b["ar_reverified_actions"]} 次。token改变不等于整段修复成功，需要阅读原文。','',
        '训练：从原始Dream重新初始化rank8 LoRA，96次更新；每32次更新由当前权重重新生成16条轨迹，共48条（24 GSM训练prompt、24新通用prompt）。轨迹没有金标准回答前缀，使用原生FlashDLM缓存及AR引导，收集时关闭重复控制器。四个连续真实输入在反传时重算dense中层特征，因此仍存在dense特征与原生缓存内部状态差异。','',
        '辅助loss=.1多标签平衡风险BCE+.1风险加权softplus(重复边界logit-AR替代logit+.5)。标签是后续最多4个外层步骤产生的启发式重复片段；修复替代来自固定AR的条件分布，并非正确答案保证。普通扩散CE另外使用192个GSM训练样本。','',
        '正式推理由启发式片段检测与CNN风险共同触发。遮盖后出现的重复片段及其后缀，重新取得Dream预测，显式选取AR top64中排除原重复边界token后的最高排名允许候选并提交，再重建两侧缓存/验证后续。EOS仅在AR原本top1时允许用于修复。每回答最多8次、每类型起点一次。此方法包含风险门控的候选排除，不是仅增加loss；原生基线首候选不匹配仍提交Dream的行为保持不变。','',
        '正常数字和词内BPE碰撞不作异常重复标签，但原始同ID计数仍全部报告；关键词意图保护不完整，指定次数的部分措辞未被识别；不能将控制组表现归因于CNN理解指令。句子相似度只是词面启发式，不是语义盲评。检测窗口、句子切分和合法复述仍可能有漏报/误报。','',
        '阈值仅用16条开发prompt选择，正式完整方法输出生成后不得调整。只有原始与完整两组，不能归因于任何单个模块。数学采用####、boxed优先再末尾数字的抽取匹配，不等同于完整推理正确性。全部原文和动作见paired-cases.json、eval/。']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.tmp') and p.name not in ('manifest.json','pipeline.log','status.json','events.jsonl')]
    save('manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files})
    with zipfile.ZipFile(ROOT/'multiscale-evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files+[ROOT/'manifest.json']:z.write(p,p.relative_to(ROOT))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
