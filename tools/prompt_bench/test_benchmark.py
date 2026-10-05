import tempfile
import unittest
from pathlib import Path

from core import BASE, BudgetExceeded, Client, Evaluator, Example, parse_label, task_prompt
from methods import choose_candidates, run_gepa
from overfit import gap_metrics, copied_spans
from leakage_check import reproduce


class FakeClient:
    workers=2
    def __init__(self): self.prompts=[]
    def complete(self,prompt,**kwargs):
        self.prompts.append(prompt)
        return dict(text='Negative' if 'not good' in prompt else 'Positive',
                    usage=dict(prompt_tokens=10,completion_tokens=1))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.pos=Example('train:1','good','Positive','sst2','train')
        self.neg=Example('train:2','not good','Negative','sst2','train')

    def test_test_data_cannot_reach_search(self):
        ev=Evaluator(FakeClient(),[self.pos],budget=10)
        target=Example('target:1','good','Positive','tweets','test')
        with self.assertRaises(ValueError): ev.evaluate(BASE,[target])
        self.assertEqual(ev.eval_count,0)

    def test_id_alias_cannot_smuggle_target(self):
        ev=Evaluator(FakeClient(),[self.pos],budget=10)
        disguised=Example(self.pos.id,'different','Positive','tweets','test')
        with self.assertRaises(ValueError): ev.evaluate(BASE,[disguised])

    def test_full_batch_budget_reservation(self):
        client=FakeClient(); ev=Evaluator(client,[self.pos,self.neg],budget=1)
        with self.assertRaises(BudgetExceeded): ev.evaluate(BASE,[self.pos,self.neg])
        self.assertEqual(client.prompts,[])
        self.assertEqual(ev.eval_count,0)

    def test_distinct_semantically_similar_inputs_are_not_reused(self):
        client=FakeClient(); ev=Evaluator(client,[self.pos,self.neg],budget=2)
        self.assertEqual([r['correct'] for r in ev.evaluate(BASE,[self.pos,self.neg])],[1,1])
        self.assertEqual(len(client.prompts),2)

    def test_exact_repeat_does_not_consume_search_budget(self):
        client=FakeClient(); ev=Evaluator(client,[self.pos],budget=1)
        ev.evaluate(BASE,[self.pos]); ev.evaluate(BASE,[self.pos])
        self.assertEqual(ev.eval_count,1)
        self.assertEqual(len(client.prompts),1)
        with self.assertRaises(BudgetExceeded): ev.evaluate('Different instruction',[self.pos])

    def test_malformed_labels_are_failures(self):
        self.assertEqual(parse_label('Positive.'),'Positive')
        for text in ['positively awful','Positive or Negative','Not Positive','']:
            self.assertEqual(parse_label(text),'INVALID')

    def test_task_schema_is_not_inferred_from_instruction(self):
        self.assertIn('Positive or Negative',task_prompt('Consider entailment.',self.pos.text))

    def test_selection_does_not_mutate_inputs(self):
        candidates=['short','longer']; values=[[1,0],[0,1]]
        a=choose_candidates(candidates,values,[self.pos,self.neg],7,'stable')
        b=choose_candidates(candidates,values,[self.pos,self.neg],7,'stable')
        self.assertEqual(a,b); self.assertEqual(values,[[1,0],[0,1]])

    def test_worst_group_rejects_single_domain(self):
        with self.assertRaises(ValueError):
            choose_candidates(['a'],[[1,1]],[self.pos,self.neg],0,'worst_group')

    def test_worst_group_can_prefer_balance(self):
        val=[Example(str(i),'x','Positive','a' if i<4 else 'b','val') for i in range(8)]
        result=choose_candidates(['uneven','balanced'],[[1,1,1,1,1,0,0,0],[1,1,1,0,1,1,1,0]],val,0,'worst_group')
        self.assertEqual(result['instruction'],'balanced')

    def test_official_gepa_adapter_offline(self):
        class ImprovingClient(FakeClient):
            def complete(self,prompt,**kwargs):
                self.prompts.append(prompt)
                if kwargs.get('temperature')==0.7:
                    text='```\nUse the improved sentiment instruction. Positive \u2192 positive.\n```'
                elif 'improved sentiment' in prompt:
                    text='Negative' if 'not good' in prompt else 'Positive'
                else:
                    text='Negative'
                return dict(text=text,usage=dict(prompt_tokens=10,completion_tokens=1))
        train=[self.pos,self.neg]
        val=[Example('val:'+e.id,e.text,e.label,e.domain,'val') for e in train]
        ev=Evaluator(ImprovingClient(),train+val,budget=80,proposal_limit=3)
        with tempfile.TemporaryDirectory() as d:
            result=run_gepa(ev,train,val,0,Path(d))
            self.assertIn('\u2192',(Path(d)/'gepa'/'run_log.txt').read_text(encoding='utf-8'))
        self.assertIn('improved',result['original_selection'])
        self.assertLessEqual(ev.eval_count,80)
        self.assertTrue(all(i.startswith(('train:','val:')) for batch in ev.audit for i in batch['example_ids']))

    def test_target_feedback_is_rejected(self):
        self.assertEqual(reproduce()['result'],'PASS')

    def test_gepa_training_scores_cannot_replace_validation(self):
        class OppositeClient(FakeClient):
            def complete(self,prompt,**kwargs):
                if kwargs.get('temperature')==0.7:
                    text='```\nImproved training instruction.\n```'
                elif 'Improved training' in prompt:
                    text='Negative' if 'not good' in prompt else 'Positive'
                else:
                    text='Negative'
                return dict(text=text,usage=dict(prompt_tokens=10,completion_tokens=1))
        train=[self.pos,self.neg]
        val=[Example('val:'+str(i),'unseen '+str(i),'Negative','sst2','val') for i in range(2)]
        ev=Evaluator(OppositeClient(),train+val,budget=80,proposal_limit=2)
        with tempfile.TemporaryDirectory() as d:
            result=run_gepa(ev,train,val,0,Path(d))
        self.assertEqual(result['vectors'],[[1,1],[0,0]])
        scores=result['official_result']['val_aggregate_scores']
        self.assertEqual(scores[0],1.)
        self.assertTrue(scores[1:] and all(s==0. for s in scores[1:]))
        self.assertEqual(result['original_selection'],BASE)

    def test_overfit_adjusts_for_baseline_split_difficulty(self):
        unchanged=gap_metrics(.98,.98,.94,.94,.90,.90)
        self.assertAlmostEqual(unchanged['raw_validation_source_gap'],.04)
        self.assertAlmostEqual(unchanged['selection_optimism'],0)
        self.assertAlmostEqual(unchanged['excess_source_gain'],0)
        overfit=gap_metrics(1,.98,.94,.94,.89,.90)
        self.assertAlmostEqual(overfit['selection_optimism'],.02)
        self.assertAlmostEqual(overfit['excess_source_gain'],.01)

    def test_copy_detection_requires_long_contiguous_match(self):
        examples=[dict(id='source:1',text='one two three four five six seven eight nine')]
        self.assertEqual(copied_spans('one two three',examples),[])
        self.assertEqual(copied_spans('Two three four five six seven eight nine.',examples)[0]['example_id'],'source:1')


if __name__=='__main__': unittest.main()
