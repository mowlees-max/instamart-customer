"""Generate the committed evaluation artifacts for the capstone."""
from pathlib import Path
import json, sys, pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.triage import run_pipeline, ZERO, FEW
from src.policy_rag import PolicyRAG, evaluate
from src.refund_agent import run_agent, resume_agent, OUT, STATE
RAW=ROOT/'data/raw'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
for checkpoint in STATE.glob('*.json'):
    checkpoint.unlink()
# Part C: zero-shot comparison and budget demonstration.
zero, zero_summary=run_pipeline(prompt=ZERO,token_budget=100000)
zero_class=zero[zero.status.isin(['classified_json','classified_keyword'])]
zero_accuracy=float((zero_class.category==pd.read_csv(RAW/'tickets.csv',keep_default_na=False).set_index('ticket_id').loc[zero_class.ticket_id,'true_category'].values).mean())
zero_tokens=int(zero.tokens_used.sum())
# run_pipeline wrote its non-final result to triage_budget_demo.csv only when token budget below 100k; preserve final few-shot output.
budget,budget_summary=run_pipeline(prompt=ZERO,token_budget=6000)
budget_exhausted=int((budget.status=='budget_exhausted').sum())
final,final_summary=run_pipeline(prompt=FEW,token_budget=100000)
final.to_csv(OUT/'triage_results.csv',index=False)
# Part D retrieval evaluation with and without overlap.
rag=PolicyRAG(40,10); overlap_eval=evaluate(rag)
no_overlap=PolicyRAG(40,0); no_eval=evaluate(no_overlap)
overlap_eval.to_csv(OUT/'policy_eval_overlap.csv',index=False); no_eval.to_csv(OUT/'policy_eval_no_overlap.csv',index=False)
# Seven refund scenarios.
sc=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False)
(OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8')
(OUT/'agent_runs.jsonl').write_text('',encoding='utf-8')
results=[]
for _,r in sc.iterrows():
    result=run_agent(r.message,r.scenario_id,run_id=f'scenario_{r.scenario_id}')
    amount=next((s['observation'].get('refund_inr') for s in result['steps'] if 'refund_inr' in s['observation']),0)
    row={'scenario_id':r.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'refund_inr':amount,'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']}
    results.append(row)
    with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
# Stop-safe demonstrations.
s1=sc.iloc[0]
token_demo=run_agent(s1.message,'S1_TOKEN_DEMO',token_limit=200,run_id='S1_TOKEN_DEMO')
time_demo=run_agent(s1.message,'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='S1_TIMEOUT_DEMO')
# Crash after second tool execution, then resume from the saved checkpoint.
try: run_agent(s1.message,'S1_CRASH_RESUME',run_id='S1_CRASH_RESUME',crash_after=2)
except RuntimeError: pass
resumed=resume_agent('S1_CRASH_RESUME')
crash_checkpoint=json.loads((STATE/'S1_CRASH_RESUME.json').read_text())
summary={
 'zero_shot':{'classified':len(zero_class),'accuracy':zero_accuracy,'tokens':zero_tokens,'tokens_per_classified':zero_tokens/max(1,len(zero_class)),'confusion':pd.crosstab(pd.read_csv(RAW/'tickets.csv',keep_default_na=False).set_index('ticket_id').loc[zero_class.ticket_id,'true_category'].values,zero_class.category).to_dict()},
 'zero_shot_budget':{'budget':6000,'budget_exhausted':budget_exhausted,'status_counts':budget.status.value_counts().to_dict()},
 'few_shot':final_summary,
 'rag_overlap':{'chunks':len(rag.chunks),'doc_hits':int(overlap_eval.loc[overlap_eval.expected_doc!='NONE','retrieved_doc'].sum()),'fact_hits':int(overlap_eval.loc[overlap_eval.expected_doc!='NONE','fact_found'].sum()),'fallbacks':int(overlap_eval.loc[overlap_eval.expected_doc=='NONE','fallback'].sum())},
 'rag_no_overlap':{'chunks':len(no_overlap.chunks),'doc_hits':int(no_eval.loc[no_eval.expected_doc!='NONE','retrieved_doc'].sum()),'fact_hits':int(no_eval.loc[no_eval.expected_doc!='NONE','fact_found'].sum()),'fallbacks':int(no_eval.loc[no_eval.expected_doc=='NONE','fallback'].sum()),'failures':no_eval.loc[(no_eval.expected_doc!='NONE')&(~no_eval.fact_found),'question'].tolist()},
 'agent_scenarios':[{k:x[k] for k in ['scenario_id','status','refund_inr','llm_calls','tokens_used']} for x in results],
 'token_demo':{k:token_demo[k] for k in ['status','llm_calls','tokens_used']},'time_demo':{k:time_demo[k] for k in ['status','llm_calls','tokens_used']},
 'crash_resume':{'status':resumed['status'],'refund_inr':next((s['observation'].get('refund_inr') for s in resumed['steps'] if 'refund_inr' in s['observation']),0),'tools':crash_checkpoint['tools_executed'],'llm_calls':resumed['llm_calls']}
}
(OUT/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
print(json.dumps(summary,indent=2,default=str))
