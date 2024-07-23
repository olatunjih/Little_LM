
**`src/model.py`**
```python
import torch
import torch.nn as nn
from transformers import LongformerModel

class EnhancedTransformerModel(nn.Module):
    def __init__(self, model_name='allenai/longformer-base-4096', num_classes=2, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()
        self.model = LongformerModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits
