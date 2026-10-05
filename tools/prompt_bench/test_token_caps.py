import unittest
from token_caps import classify, metrics, select_caps


class CapTests(unittest.TestCase):
    def test_cap_is_forwarded_and_partial_label_is_invalid(self):
        class Client:
            def complete(self,prompt,**kwargs):
                assert kwargs['max_tokens']==1
                return dict(text='Pos',finish_reason='length')
        row=classify(Client(),'selection',dict(id='source',text='good',label='Positive'),1)
        self.assertEqual(row['prediction'],'INVALID')
        self.assertEqual(row['correct'],0)

    def test_valid_label_at_limit_is_distinct_from_invalid_output(self):
        ref=[dict(id='p',label='Positive',prediction='Positive',correct=1,finish_reason='stop',
                  usage=dict(completion_tokens=2,prompt_tokens=20),seconds=1.,model='gemma')]
        limited=[dict(ref[0],finish_reason='length',usage=dict(completion_tokens=1,prompt_tokens=20))]
        a=metrics(limited,ref); b=metrics(ref,ref)
        self.assertEqual(a['invalid'],0)
        self.assertEqual(a['limit_finishes'],1)
        chosen=select_caps({1:a,2:b,16:b})
        self.assertEqual(chosen['minimum_valid_cap'],1)
        self.assertEqual(chosen['minimum_natural_stop_cap'],2)

    def test_selection_rejects_prediction_changes_and_invalids(self):
        baseline=dict(invalid=0,accuracy=1.,macro_f1=1.,matches_baseline=2,n=2,limit_finishes=0)
        lower=dict(baseline,invalid=1,accuracy=.5,matches_baseline=1)
        self.assertEqual(select_caps({1:lower,2:baseline,16:baseline})['minimum_valid_cap'],2)


if __name__=='__main__': unittest.main()
