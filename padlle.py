import paddlehub as hub
from paddle.fluid.framework import switch_main_program
import paddle.fluid as fluid
import pandas as pd
import numpy as np
import codecs
import csv
from paddlehub.dataset import InputExample, BaseDataset
class MultiMydatas(BaseDataset):
    def __init__(self):
        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = "trainSet.csv"
        self.train_examples = self._read_csv(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = "devSet.csv"
        self.dev_examples = self._read_csv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = "testSet.csv"
        self.test_examples = self._read_csv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ['难过', '开心', '赞赏', '批评', '怀疑', '喜欢', '惊叹', '鼓励']

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_csv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = pd.read_csv(input_file, sep='\t', encoding="UTF-8")
        examples = []
        for index, row in data.iterrows():
            guid = row["id"]
            text = row["text_a"]
            labels = [int(value) for value in row[2:]]
            example = InputExample(guid=guid, label=labels, text_a=text)
            examples.append(example)

        return examples


dataset = MultiMydatas()
module = hub.Module(name='ernie')
reader = hub.reader.MultiLabelClassifyReader(dataset=dataset, vocab_path=module.get_vocab_path(), max_seq_len=128)

strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5,
    lr_scheduler='linear_decay',
    optimizer_name='adam'
)

config = hub.RunConfig(
    use_data_parallel=False,
    use_pyreader=False,
    use_cuda=False,
    batch_size=32,
    enable_memory_optim=False,
    checkpoint_dir='ernie_txt_cls_turtorial_demo',
    num_epoch=100,
    strategy=strategy,
)
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
pooled_output = outputs['pooled_output']
feed_list = [
    inputs['input_ids'].name,
    inputs['position_ids'].name,
    inputs['segment_ids'].name,
    inputs['input_mask'].name,
]

cls_task = hub.MultiLabelClassifierTask(data_reader=reader,
                                        feature=pooled_output,
                                        feed_list=feed_list,
                                        num_classes=dataset.num_labels,
                                        config=config)
run_states = cls_task.finetune_and_eval()
import numpy as np
print(list(pooled_output))
# da = list(pd.read_csv('datase.csv')['评论'])
# data=[]
# for i in da:
#     data.append([i])
# run_states = cls_task.predict(data=data,return_result=True)
# print(run_states)
# sizess={'难过':0, '开心':0, '赞赏':0, '批评':0, '怀疑':0, '喜欢':0, '惊叹':0, '鼓励':0,'其他':0}
# for i in run_states:
#     ti=0
#     for j in i:
#         for k,v in j.items():
#             if v==1:
#                 sizess[k]+=1
#             else:
#                 ti+=1
#         if ti==8:
#             sizess['其他']+=1

# labels = ['难过', '开心', '赞赏', '批评', '怀疑', '喜欢', '惊叹', '鼓励','其他']
# sizes = sizess.values()
# print(sizes)