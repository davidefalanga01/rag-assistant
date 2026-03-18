from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

'''Clean the generated RAG evaluation dataset by removing question-context pairs with low-quality contexts.'''
import json

json_path = "data/eval_rag_dataset.json"
dataset = EmbeddingQAFinetuneDataset.from_json(json_path)
print(dataset['query'][0])
# def clean_dataset(dataset_path, output_path, quality_threshold=0.5):
#     with open(dataset_path, 'r') as f:
#         data = json.load(f)

#     cleaned_data = []
#     for item in data:
#         if item.get('context_quality', 0) >= quality_threshold:
#             cleaned_data.append(item)

#     with open(output_path, 'w') as f:
#         json.dump(cleaned_data, f)